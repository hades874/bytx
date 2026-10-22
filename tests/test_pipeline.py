from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from bytx import fetch, ledger, models, pipeline, platforms, store
from bytx.config import (
    AsrConfig,
    AsrModelConfig,
    Config,
    FetchConfig,
    OllamaConfig,
    PathsConfig,
    RagConfig,
    SummarizeConfig,
)
from bytx.transcribe import TranscribeResult, TranscribeStats


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    store.init_db(c)
    yield c
    c.close()


def make_config(tmp_path: Path, *, summarize_enabled: bool = True) -> Config:
    return Config(
        root=tmp_path,
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            transcribed_files_dir=tmp_path / "yt-transcribed-files",
        ),
        asr=AsrConfig(
            model="m", device="auto", compute_type="float16", beam_size=1,
            vad_filter=False, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
            repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="",
            models={
                # "m" is a Bangla-only model, "e" a multilingual one (English).
                "m": AsrModelConfig(key="m", source="org/m", language="bn"),
                "e": AsrModelConfig(key="e", source="org/e", language=""),
            },
            language_models={"bn": "m", "en": "e"},
        ),
        fetch=FetchConfig(
            force_asr=False, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""
        ),
        ollama=OllamaConfig(host="http://x", embed_model="bge-m3", chat_model="gemma3:4b", num_ctx=8192),
        rag=RagConfig(chunk_seconds=75, chunk_overlap_seconds=15, top_k=8),
        summarize=SummarizeConfig(enabled=summarize_enabled, map_window_minutes=10, language="auto"),
    )


class FakeModel:
    """One model handed out by FakePool. Per-video results/errors are looked up
    by audio path stem (~video id)."""

    def __init__(self, pool: FakePool, model_cfg: AsrModelConfig):
        self.pool = pool
        self.model_cfg = model_cfg

    def sample_text(self, audio_path: Path, language: str | None = None) -> str:
        self.pool.sampled.append((self.model_cfg.key, audio_path.stem, language))
        if audio_path.stem in _SAMPLE_FAILURES:
            raise RuntimeError(_SAMPLE_FAILURES[audio_path.stem])
        return _SAMPLE_TEXT.get(audio_path.stem, "")

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscribeResult:
        self.pool.transcribed.append((self.model_cfg.key, audio_path, language))
        if audio_path.stem in _TRANSCRIBE_FAILURES:
            raise RuntimeError(_TRANSCRIBE_FAILURES[audio_path.stem])
        segs = _TRANSCRIBE_RESULTS.get(
            audio_path.stem, [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="default text")]
        )
        return TranscribeResult(
            segments=segs, device="cuda", compute_type="float16", language=language,
            stats=_TRANSCRIBE_STATS.get(audio_path.stem, TranscribeStats(speech_seconds=60.0)),
        )


class FakePool:
    """Replaces bytx.pipeline.TranscriberPool. Records which model was loaded
    when (like the real pool, only one at a time), plus transcribe calls."""

    instances: list[FakePool] = []

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.current: str | None = None
        self.loads: list[str] = []  # model keys, in the order they were loaded
        self.transcribed: list[tuple[str, Path, str | None]] = []  # (model key, audio, language)
        self.sampled: list[tuple[str, str, str | None]] = []  # (model key, video id, language)
        self.unloaded = False
        FakePool.instances.append(self)

    def _switch(self, model_cfg: AsrModelConfig) -> FakeModel:
        if self.current != model_cfg.key:
            self.current = model_cfg.key
            self.loads.append(model_cfg.key)
        return FakeModel(self, model_cfg)

    def for_language(self, language: str) -> FakeModel:
        return self._switch(self.cfg.asr.model_for_language(language))

    def unload(self) -> None:
        self.unloaded = True
        self.current = None


_TRANSCRIBE_RESULTS: dict[str, list[store.Segment]] = {}
_TRANSCRIBE_FAILURES: dict[str, str] = {}
_TRANSCRIBE_STATS: dict[str, TranscribeStats] = {}
_SAMPLE_TEXT: dict[str, str] = {}  # video id -> text the Bangla model writes for its sampled audio
_SAMPLE_FAILURES: dict[str, str] = {}


class FakeOllamaClient:
    """Replaces bytx.pipeline.OllamaClient as a context manager."""

    instances: list[FakeOllamaClient] = []
    reachable = True
    pulled: set[str] | None = None  # None: every model is pulled

    def __init__(self, ollama_cfg):
        self.cfg = ollama_cfg
        self.closed = False
        FakeOllamaClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def is_reachable(self):
        return FakeOllamaClient.reachable

    def has_model(self, model):
        return FakeOllamaClient.pulled is None or model in FakeOllamaClient.pulled

    def embed(self, texts, model=None):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def chat(self, messages, model=None, num_ctx=None, temperature=None):
        return "সারাংশ পাঠ্য"


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakePool.instances = []
    FakeOllamaClient.instances = []
    FakeOllamaClient.reachable = True
    FakeOllamaClient.pulled = None
    _TRANSCRIBE_RESULTS.clear()
    _TRANSCRIBE_FAILURES.clear()
    _TRANSCRIBE_STATS.clear()
    _SAMPLE_TEXT.clear()
    _SAMPLE_FAILURES.clear()
    monkeypatch.setattr(pipeline, "TranscriberPool", FakePool)
    monkeypatch.setattr(pipeline, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(pipeline.models, "missing", lambda cfg: [])  # `bytx setup` has run
    yield


def patch_fetch(monkeypatch, tmp_path, *, human_subs: dict[str, list[store.Segment]] | None = None,
                 info_failures: set[str] | None = None,
                 human_sub_lang: str = "bn"):
    """Wires fetch.get_video_info / download_audio / fetch_human_subtitle_segments
    to deterministic fakes based on the video id extracted from the URL.
    `human_subs` videos get one uploader subtitle track in `human_sub_lang`.
    Returns the list of video ids whose audio gets downloaded, in order.
    """
    human_subs = human_subs or {}
    info_failures = info_failures or set()
    downloads: list[str] = []

    def fake_get_video_info(url, cfg):
        vid = platforms.parse_video_ref(url).id
        if vid in info_failures:
            raise RuntimeError(f"metadata fetch failed for {vid}")
        return fetch.VideoInfo(
            id=vid, url=url, title=f"Title {vid}", channel="Chan", channel_id="c1",
            upload_date="20260101", duration=60.0, description="desc",
            human_subtitles=({human_sub_lang: f"http://subs/{vid}.vtt"} if vid in human_subs else {}),
        )

    def fake_download_audio(url, out_dir, cfg):
        vid = platforms.parse_video_ref(url).id
        downloads.append(vid)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{vid}.webm"
        p.write_bytes(b"fake audio")
        return p

    def fake_fetch_human_subtitle_segments(sub_url):
        vid = sub_url.split("/")[-1].split(".")[0]
        return human_subs[vid]

    monkeypatch.setattr(pipeline.fetch, "get_video_info", fake_get_video_info)
    monkeypatch.setattr(pipeline.fetch, "download_audio", fake_download_audio)
    monkeypatch.setattr(pipeline.fetch, "fetch_human_subtitle_segments", fake_fetch_human_subtitle_segments)
    return downloads


def video_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


# --- add_targets ------------------------------------------------------------

def test_add_targets_inserts_new_queued_videos(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    urls = [video_url("aaaaaaaaaaa"), video_url("bbbbbbbbbbb")]
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: urls)

    new_ids = pipeline.add_targets(conn, cfg, ["some target"])
    assert set(new_ids) == {"aaaaaaaaaaa", "bbbbbbbbbbb"}
    videos = store.list_videos(conn)
    assert {v.id for v in videos} == {"aaaaaaaaaaa", "bbbbbbbbbbb"}
    assert all(v.status == "queued" for v in videos)


def test_add_targets_skips_already_known_ids(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    store.upsert_video(conn, store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), status="transcribed"))
    urls = [video_url("aaaaaaaaaaa"), video_url("bbbbbbbbbbb")]
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: urls)

    new_ids = pipeline.add_targets(conn, cfg, ["some target"])
    assert new_ids == ["bbbbbbbbbbb"]
    # existing video's status must be untouched
    assert store.get_video(conn, "aaaaaaaaaaa").status == "transcribed"


def test_add_targets_skips_ids_only_in_ledger_when_db_empty(conn, tmp_path, monkeypatch):
    """Recovery scenario: data/ (and its SQLite DB) was wiped or this is a
    fresh clone, but TRANSCRIBED_LEDGER.md survived (it's git-tracked)."""
    cfg = make_config(tmp_path)
    ledger.append_entry(
        ledger.ledger_path(cfg),
        store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), channel="C", title="T"),
    )
    urls = [video_url("aaaaaaaaaaa"), video_url("bbbbbbbbbbb")]
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: urls)

    new_ids = pipeline.add_targets(conn, cfg, ["some target"])
    assert new_ids == ["bbbbbbbbbbb"]
    assert store.get_video(conn, "aaaaaaaaaaa") is None


# --- transcribe_one -----------------------------------------------------------

def test_transcribe_one_warns_when_clips_still_truncated(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_STATS[vid] = TranscribeStats(truncated_unresolved=2, speech_seconds=60.0)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    assert "token limit" in caplog.text
    assert store.get_video(conn, vid).status == "transcribed"  # a warning, not a failure


def test_transcribe_one_warns_when_little_speech_detected(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "ccccccccccc"
    patch_fetch(monkeypatch, tmp_path)  # fake video duration is 60s
    _TRANSCRIBE_STATS[vid] = TranscribeStats(speech_seconds=30.0)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    assert "detected as speech" in caplog.text


def test_transcribe_one_no_warning_for_healthy_asr(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "ddddddddddd"
    patch_fetch(monkeypatch, tmp_path)  # fake video duration is 60s
    # A minute of ordinary speech: ~130 words spanning the whole video.
    _TRANSCRIBE_RESULTS[vid] = [
        store.Segment(idx=i, start_s=i * 30.0, end_s=(i + 1) * 30.0, text=" ".join(["শব্দ"] * 65))
        for i in range(2)
    ]
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    assert caplog.text == ""


def test_transcribe_one_logs_quality_flags_for_thin_output(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "eeeeeeeeeee"
    patch_fetch(monkeypatch, tmp_path)  # 60 s video; the default fake text is 2 words
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    assert "words/min" in caplog.text
    assert store.get_video(conn, vid).status == "transcribed"


def test_transcribe_one_prefers_human_subs(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="মানুষের সাবটাইটেল")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs})
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)

    pool = FakePool(cfg)
    pipeline.transcribe_one(conn, cfg, pool, video)

    assert pool.transcribed == []  # ASR must not run when human subs exist
    stored = store.get_segments(conn, vid)
    assert [s.text for s in stored] == ["মানুষের সাবটাইটেল"]
    updated = store.get_video(conn, vid)
    assert updated.source == "human_subs"
    assert updated.asr_model is None
    assert updated.status == "transcribed"

    out_dir = cfg.paths.videos_dir / vid
    assert (out_dir / "transcript.txt").exists()
    assert (out_dir / "info.json").exists()


def test_transcribe_one_writes_human_facing_copy_and_ledger_entry(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="মানুষের সাবটাইটেল")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs})
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)

    pool = FakePool(cfg)
    pipeline.transcribe_one(conn, cfg, pool, video)

    out_dir = cfg.paths.transcribed_files_dir / date.today().isoformat() / f"Chan - Title {vid}"
    transcript = out_dir / "transcript.txt"
    assert transcript.exists()
    assert transcript.read_text(encoding="utf-8") == "মানুষের সাবটাইটেল\n"
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "Title " + vid and meta["source"] == "human_subs"
    assert meta["transcribed_at"]
    assert (out_dir / "transcript.vtt").exists() and (out_dir / "transcript.json").exists()
    assert vid in (cfg.paths.transcribed_files_dir / "INDEX.md").read_text(encoding="utf-8")

    ledger_content = ledger.ledger_path(cfg).read_text(encoding="utf-8")
    assert f"- id: {vid}" in ledger_content


def test_transcribe_one_uses_asr_and_deletes_audio(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_RESULTS[vid] = [store.Segment(idx=0, start_s=0.0, end_s=3.0, text="এএসআর ফলাফল")]
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)

    pool = FakePool(cfg)
    pipeline.transcribe_one(conn, cfg, pool, video)

    assert len(pool.transcribed) == 1
    _key, audio_path, _language = pool.transcribed[0]
    assert not audio_path.exists()  # deleted after transcription

    stored = store.get_segments(conn, vid)
    assert [s.text for s in stored] == ["এএসআর ফলাফল"]
    updated = store.get_video(conn, vid)
    assert updated.source == "asr"
    assert updated.asr_model == "m"


def test_transcribe_one_deletes_audio_even_on_transcribe_failure(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "ccccccccccc"
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_FAILURES[vid] = "boom"
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)
    pool = FakePool(cfg)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.transcribe_one(conn, cfg, pool, video)

    audio_path = cfg.paths.audio_dir / f"{vid}.webm"
    assert not audio_path.exists()


# --- index_one / summarize_one -------------------------------------------------

def test_index_one_stores_chunks_and_sets_status(conn, tmp_path):
    cfg = make_config(tmp_path)
    vid = "ddddddddddd"
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="transcribed"))
    store.replace_segments(conn, vid, [store.Segment(idx=0, start_s=0, end_s=10, text="কিছু লেখা")])
    video = store.get_video(conn, vid)

    with FakeOllamaClient(cfg.ollama) as client:
        pipeline.index_one(conn, cfg, client, video)

    assert len(store.get_chunks(conn, vid)) == 1
    assert store.get_video(conn, vid).status == "indexed"


def test_summarize_one_writes_summary_and_file(conn, tmp_path):
    cfg = make_config(tmp_path)
    vid = "eeeeeeeeeee"
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), title="T", status="indexed"))
    store.replace_segments(conn, vid, [store.Segment(idx=0, start_s=0, end_s=10, text="কিছু লেখা")])
    video = store.get_video(conn, vid)

    with FakeOllamaClient(cfg.ollama) as client:
        pipeline.summarize_one(conn, cfg, client, video)

    assert store.get_summary(conn, vid) == "সারাংশ পাঠ্য"
    assert store.get_video(conn, vid).status == "summarized"
    assert (cfg.paths.videos_dir / vid / "summary.md").exists()


# --- run() end-to-end ----------------------------------------------------------

def test_run_happy_path_all_stages(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vids = ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    patch_fetch(monkeypatch, tmp_path)
    for v in vids:
        store.upsert_video(conn, store.Video(id=v, url=video_url(v), status="queued"))

    result = pipeline.run(conn, cfg)

    assert result.transcribed == 2
    assert result.indexed == 2
    assert result.summarized == 2
    assert result.failed == 0
    for v in vids:
        assert store.get_video(conn, v).status == "summarized"
    assert FakePool.instances[0].unloaded is True


def test_run_before_setup_stops_and_leaves_every_video_queued(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    downloads = patch_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline.models, "missing", lambda cfg: ["m", "e"])
    vids = ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    for v in vids:
        store.upsert_video(conn, store.Video(id=v, url=video_url(v), status="queued"))

    with pytest.raises(models.ModelsMissing, match="Run `bytx setup`"):
        pipeline.run(conn, cfg)

    assert [store.get_video(conn, v).status for v in vids] == ["queued", "queued"]
    assert downloads == [] and FakePool.instances == []


def test_run_with_an_empty_queue_does_not_need_the_asr_models(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(pipeline.models, "missing", lambda cfg: ["m"])
    assert pipeline.run(conn, cfg).failed == 0  # indexing/summaries of old videos can still run


def test_run_without_ollama_stops_after_transcription_without_failing(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    patch_fetch(monkeypatch, tmp_path)
    store.upsert_video(conn, store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), status="queued"))
    FakeOllamaClient.reachable = False

    result = pipeline.run(conn, cfg)

    assert (result.transcribed, result.indexed, result.failed) == (1, 0, 0)
    assert "Ollama (optional) isn't running" in result.stopped
    assert store.get_video(conn, "aaaaaaaaaaa").status == "transcribed"

    FakeOllamaClient.reachable = True  # the user starts Ollama and runs again
    again = pipeline.run(conn, cfg)

    assert (again.indexed, again.summarized, again.stopped) == (1, 1, None)
    assert store.get_video(conn, "aaaaaaaaaaa").status == "summarized"


def test_run_with_ollama_models_not_pulled_says_run_setup(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    store.upsert_video(conn, store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), status="transcribed"))
    store.replace_segments(conn, "aaaaaaaaaaa", [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="t")])
    FakeOllamaClient.pulled = {"gemma3:4b"}  # no embedding model

    result = pipeline.run(conn, cfg)

    assert "bge-m3" in result.stopped and "bytx setup" in result.stopped
    assert store.get_video(conn, "aaaaaaaaaaa").status == "transcribed"


def test_run_without_summaries_does_not_need_the_chat_model(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path, summarize_enabled=False)
    store.upsert_video(conn, store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), status="transcribed"))
    store.replace_segments(conn, "aaaaaaaaaaa", [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="t")])
    FakeOllamaClient.pulled = {"bge-m3"}

    result = pipeline.run(conn, cfg)

    assert (result.indexed, result.stopped) == (1, None)


def test_run_with_nothing_to_index_never_contacts_ollama(conn, tmp_path):
    assert pipeline.run(conn, make_config(tmp_path)).stopped is None
    assert FakeOllamaClient.instances == []


def test_run_continues_after_one_video_fails(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    good, bad = "aaaaaaaaaaa", "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path, info_failures={bad})
    for v in (good, bad):
        store.upsert_video(conn, store.Video(id=v, url=video_url(v), status="queued"))

    result = pipeline.run(conn, cfg)

    assert result.failed == 1
    assert store.get_video(conn, bad).status == "failed"
    assert "metadata fetch failed" in store.get_video(conn, bad).error
    assert store.get_video(conn, bad).attempts == 1
    # the good video should still make it all the way through
    assert store.get_video(conn, good).status == "summarized"


def test_run_respects_limit(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vids = ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    patch_fetch(monkeypatch, tmp_path)
    for v in vids:
        store.upsert_video(conn, store.Video(id=v, url=video_url(v), status="queued"))

    result = pipeline.run(conn, cfg, limit=2)

    assert result.transcribed == 2
    remaining_queued = store.list_videos(conn, status="queued")
    assert len(remaining_queued) == 1


def test_run_without_summarize_stops_at_indexed(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path, summarize_enabled=False)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    result = pipeline.run(conn, cfg)

    assert result.summarized == 0
    assert store.get_video(conn, vid).status == "indexed"
    assert store.get_summary(conn, vid) is None


def test_run_summarize_enabled_override_beats_config(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path, summarize_enabled=True)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    result = pipeline.run(conn, cfg, summarize_enabled=False)

    assert result.summarized == 0
    assert store.get_video(conn, vid).status == "indexed"


def test_run_resumes_without_reloading_asr_model_when_nothing_queued(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="transcribed"))
    store.replace_segments(conn, vid, [store.Segment(idx=0, start_s=0, end_s=1, text="আগের ফলাফল")])

    result = pipeline.run(conn, cfg)

    assert FakePool.instances == []  # never constructed: no queued work
    assert result.indexed == 1
    assert store.get_video(conn, vid).status == "summarized"


# --- retry_failed --------------------------------------------------------------

def test_retry_failed_requeues_and_clears_error(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1", status="failed", error="boom", attempts=2))
    store.upsert_video(conn, store.Video(id="v2", url="u2", status="transcribed"))

    count = pipeline.retry_failed(conn)

    assert count == 1
    v1 = store.get_video(conn, "v1")
    assert v1.status == "queued"
    assert v1.error is None
    assert v1.attempts == 2  # history preserved
    assert store.get_video(conn, "v2").status == "transcribed"  # untouched


def _failed_video(conn, vid: str, *, segments: bool = False, chunks: bool = False) -> None:
    store.upsert_video(conn, store.Video(id=vid, url=f"u-{vid}", status="failed", error="boom"))
    if segments:
        store.replace_segments(conn, vid, [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="t")])
    if chunks:
        store.replace_chunks(
            conn, vid,
            [store.Chunk(id=None, video_id=vid, start_s=0.0, end_s=1.0, text="t",
                         embedding=store.floats_to_blob([0.1]), embed_model="m")],
        )


def test_retry_failed_resumes_at_earliest_unfinished_stage(conn):
    _failed_video(conn, "v1")
    _failed_video(conn, "v2", segments=True)
    _failed_video(conn, "v3", segments=True, chunks=True)

    assert pipeline.retry_failed(conn) == 3

    assert store.get_video(conn, "v1").status == "queued"
    assert store.get_video(conn, "v2").status == "transcribed"
    assert store.get_video(conn, "v3").status == "indexed"


@pytest.mark.parametrize("target", ["write_transcript_copy", "append_entry"])
def test_transcribe_one_extras_failure_does_not_fail_video(conn, tmp_path, monkeypatch, target):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="সাবটাইটেল")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs})
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    def boom(*args, **kwargs):
        raise OSError("locked")

    module = pipeline.export if target == "write_transcript_copy" else pipeline.ledger
    monkeypatch.setattr(module, target, boom)

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    assert store.get_video(conn, vid).status == "transcribed"


# --- languages -------------------------------------------------------------------

def _queue(conn, vid: str, language: str | None = None) -> store.Video:
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued", language=language))
    return store.get_video(conn, vid)


def test_add_targets_stores_the_requested_language(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [video_url("aaaaaaaaaaa")])

    pipeline.add_targets(conn, cfg, ["t"], language="en")

    assert store.get_video(conn, "aaaaaaaaaaa").language == "en"


def test_add_targets_defaults_to_the_configured_language(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [video_url("aaaaaaaaaaa")])

    pipeline.add_targets(conn, cfg, ["t"])

    assert store.get_video(conn, "aaaaaaaaaaa").language == "bn"


def test_add_targets_default_language_follows_the_config(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(cfg, asr=dataclasses.replace(cfg.asr, default_language="en"))
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [video_url("aaaaaaaaaaa")])

    pipeline.add_targets(conn, cfg, ["t"])

    assert store.get_video(conn, "aaaaaaaaaaa").language == "en"


def test_english_video_uses_the_english_model(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid, language="en"))

    assert [(key, lang) for key, _audio, lang in pool.transcribed] == [("e", "en")]
    updated = store.get_video(conn, vid)
    assert (updated.language, updated.source, updated.asr_model) == ("en", "asr", "e")


def test_bangla_video_uses_the_bangla_model(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid, language="bn"))

    assert [(key, lang) for key, _audio, lang in pool.transcribed] == [("m", "bn")]
    assert store.get_video(conn, vid).asr_model == "m"


def test_video_queued_before_language_support_is_treated_as_the_default_language(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid))  # language NULL, as in an old queue

    assert [(key, lang) for key, _audio, lang in pool.transcribed] == [("m", "bn")]
    assert store.get_video(conn, vid).language == "bn"


def test_youtube_metadata_never_overrides_the_chosen_language(conn, tmp_path, monkeypatch):
    """Regression: YouTube tags real Bangla videos in this library en-US. The video
    must stay Bangla (the default) whatever the metadata claims."""
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    monkeypatch.setattr(
        fetch, "get_video_info",
        lambda url, cfg: fetch.VideoInfo(
            id=vid, url=url, title="t", channel="c", channel_id="c1", upload_date="20260101",
            duration=60.0, description="d",
        ),
    )
    monkeypatch.setattr(pipeline.fetch, "download_audio", lambda url, out_dir, cfg: _fake_audio(out_dir, vid))
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid))

    assert pool.transcribed[0][0] == "m"


def _fake_audio(out_dir, vid):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{vid}.webm"
    path.write_bytes(b"fake audio")
    return path


def test_human_subs_in_the_videos_language_are_used_without_downloading_audio(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="uploader captions")]
    downloads = patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs}, human_sub_lang="en")
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid, language="en"))

    assert downloads == []
    assert pool.transcribed == []
    assert [s.text for s in store.get_segments(conn, vid)] == ["uploader captions"]
    assert store.get_video(conn, vid).source == "human_subs"


def test_human_subs_in_another_language_are_not_used(conn, tmp_path, monkeypatch):
    """A Bangla video that only carries English subtitles (a translation) must
    be transcribed from its audio, not have the English subs stored as its text."""
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="english translation")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs}, human_sub_lang="en")
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid, language="bn"))

    assert [(key, lang) for key, _audio, lang in pool.transcribed] == [("m", "bn")]
    assert store.get_video(conn, vid).source == "asr"


def test_force_asr_ignores_human_subs_for_any_language(conn, tmp_path, monkeypatch):
    cfg = dataclasses.replace(
        make_config(tmp_path),
        fetch=FetchConfig(force_asr=True, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""),
    )
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="uploader captions")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs}, human_sub_lang="en")
    pool = FakePool(cfg)

    pipeline.transcribe_one(conn, cfg, pool, _queue(conn, vid, language="en"))

    assert len(pool.transcribed) == 1
    assert store.get_video(conn, vid).source == "asr"


def test_unmapped_language_fails_before_downloading_audio(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    downloads = patch_fetch(monkeypatch, tmp_path)
    video = _queue(conn, "aaaaaaaaaaa", language="fr")

    with pytest.raises(ValueError, match="No ASR model is configured for language 'fr'"):
        pipeline.transcribe_one(conn, cfg, FakePool(cfg), video)

    assert downloads == []


def test_run_fails_unsupported_language_but_continues_queue(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    bad, good = "aaaaaaaaaaa", "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path)
    _queue(conn, bad, language="fr")
    _queue(conn, good, language="bn")

    result = pipeline.run(conn, cfg)

    assert result.failed == 1
    assert "language 'fr'" in store.get_video(conn, bad).error
    assert store.get_video(conn, good).status == "summarized"


def test_run_groups_videos_by_language_to_minimise_model_loads(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    order = [("aaaaaaaaaaa", "en"), ("bbbbbbbbbbb", "bn"), ("ccccccccccc", "en"), ("ddddddddddd", "bn")]
    patch_fetch(monkeypatch, tmp_path)
    for vid, lang in order:
        _queue(conn, vid, language=lang)

    result = pipeline.run(conn, cfg)

    assert result.transcribed == 4
    pool = FakePool.instances[0]
    assert pool.loads == ["m", "e"]  # each model loaded once, not once per switch
    assert [audio.stem for _k, audio, _l in pool.transcribed] == [
        "bbbbbbbbbbb", "ddddddddddd", "aaaaaaaaaaa", "ccccccccccc",
    ]
    assert pool.unloaded is True


def test_run_puts_videos_without_a_language_in_the_default_language_group(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    patch_fetch(monkeypatch, tmp_path)
    _queue(conn, "aaaaaaaaaaa", language="en")
    _queue(conn, "bbbbbbbbbbb")  # language NULL -> bn

    pipeline.run(conn, cfg)

    assert FakePool.instances[0].loads == ["m", "e"]


def test_language_survives_refresh_of_metadata(conn, tmp_path, monkeypatch):
    """transcribe_one upserts fresh metadata (language=None) over the row; that
    must not wipe a language chosen earlier, e.g. by --lang."""
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    video = _queue(conn, vid, language="en")

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), video)

    assert store.get_video(conn, vid).language == "en"


def test_language_recorded_in_ledger_and_json_exports(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), _queue(conn, vid, language="en"))

    assert "  language: en\n" in ledger.ledger_path(cfg).read_text(encoding="utf-8")
    out_dir = cfg.paths.videos_dir / vid
    assert json.loads((out_dir / "info.json").read_text(encoding="utf-8"))["language"] == "en"
    assert json.loads((out_dir / "transcript.json").read_text(encoding="utf-8"))["language"] == "en"


# --- --lang auto (language detection from a transcribed sample) ---------------------------

BANGLISH_SAMPLE = (
    "আপনিরা যারা হচ্ছে বাংলাদেশে ব্যাচেলর কম্পিট করেছেন এবং হচ্ছে বিদেশে আসতে যাচ্ছেন "
    "তাদের জন্য এই ভিডিওটা কারণ আমি নিজেও এসেছি এবং আমার অনেক কিছু মনে হয়েছে যেটা আপনাদের বলা দরকার "
) * 3
ENGLISH_SAMPLE = (
    "আলার্ভ রিভান্ড বালকাম টু কথার স্লোক আই হোপ ইউ অল আর ডুইং আমেইজিং সো "
    "হোয়েন আই ওয়াজ ইন হাই স্কুল আই ওয়াজ ভেরি ইন্টারেস্টেড ইন হাউ থিংস ওয়ার্ক "
) * 3


def _samples(monkeypatch, **by_video: str) -> None:
    """What the Bangla model 'writes' for each auto video's sampled audio."""
    _SAMPLE_TEXT.clear()
    _SAMPLE_TEXT.update(by_video)


def test_add_targets_auto_leaves_the_language_undecided(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [video_url("aaaaaaaaaaa")])

    pipeline.add_targets(conn, cfg, ["t"], language=pipeline.AUTO)

    assert store.get_video(conn, "aaaaaaaaaaa").language is None


def test_auto_video_with_bangla_speech_stays_bangla_and_reuses_the_audio(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    downloads = patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{vid: BANGLISH_SAMPLE})
    _queue(conn, vid)

    result = pipeline.run(conn, cfg)

    assert (result.detected, result.transcribed, result.failed) == (1, 1, 0)
    updated = store.get_video(conn, vid)
    assert (updated.language, updated.asr_model, updated.source) == ("bn", "m", "asr")
    assert downloads == [vid]  # detection and transcription shared one download
    assert not (cfg.paths.audio_dir / f"{vid}.webm").exists()


def test_auto_video_with_english_speech_goes_to_the_english_model(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    downloads = patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{vid: ENGLISH_SAMPLE})
    _queue(conn, vid)

    pipeline.run(conn, cfg)

    updated = store.get_video(conn, vid)
    assert (updated.language, updated.asr_model) == ("en", "e")
    pool = FakePool.instances[0]
    assert [(k, lang) for k, _a, lang in pool.transcribed] == [("e", "en")]
    assert downloads == [vid]


def test_auto_detection_always_samples_with_the_bangla_model(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    cfg = dataclasses.replace(cfg, asr=dataclasses.replace(cfg.asr, default_language="en"))
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{vid: BANGLISH_SAMPLE})
    _queue(conn, vid)

    pipeline.run(conn, cfg)

    pool = FakePool.instances[0]
    assert pool.sampled == [("m", vid, "bn")]  # what the Bangla model writes is the signal
    assert store.get_video(conn, vid).language == "bn"


def test_run_mixes_known_and_auto_videos_without_extra_model_loads(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    known_en, auto_en, auto_bn = "aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"
    patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{auto_en: ENGLISH_SAMPLE, auto_bn: BANGLISH_SAMPLE})
    _queue(conn, known_en, language="en")
    _queue(conn, auto_en)
    _queue(conn, auto_bn)

    result = pipeline.run(conn, cfg)

    assert (result.detected, result.transcribed) == (2, 3)
    pool = FakePool.instances[0]
    assert pool.loads == ["m", "e"]  # detection used the Bangla model, then bn group, then en group
    assert [s for s in (store.get_video(conn, v).language for v in (known_en, auto_en, auto_bn))] == ["en", "en", "bn"]


def test_auto_with_too_little_speech_uses_the_default_language_and_says_so(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{vid: "হ্যালো এভরিওয়ান"})
    _queue(conn, vid)

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.run(conn, cfg)

    assert store.get_video(conn, vid).language == "bn"
    assert "too little speech" in caplog.text
    assert "--lang" in caplog.text


def test_auto_detection_failure_fails_only_that_video_and_cleans_up(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    bad, good = "aaaaaaaaaaa", "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{good: BANGLISH_SAMPLE})
    _SAMPLE_FAILURES[bad] = "sampling exploded"
    _queue(conn, bad)
    _queue(conn, good)

    result = pipeline.run(conn, cfg)

    assert (result.failed, result.transcribed) == (1, 1)
    assert store.get_video(conn, bad).status == "failed"
    assert "sampling exploded" in store.get_video(conn, bad).error
    assert store.get_video(conn, good).status == "summarized"
    assert not list(cfg.paths.audio_dir.glob("*.webm"))  # neither video left audio behind


def test_downloaded_audio_is_removed_when_the_transcription_stage_fails_early(conn, tmp_path, monkeypatch):
    """The audio fetched for detection must not leak if the metadata fetch (the
    first thing transcribe_one does) fails before the audio is ever used."""
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path, info_failures={vid})
    _samples(monkeypatch, **{vid: BANGLISH_SAMPLE})
    _queue(conn, vid)

    result = pipeline.run(conn, cfg)

    assert result.failed == 1
    assert not list(cfg.paths.audio_dir.glob("*.webm"))


def test_auto_video_with_human_subs_in_the_detected_language_uses_them(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="uploader captions")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs}, human_sub_lang="en")
    _samples(monkeypatch, **{vid: ENGLISH_SAMPLE})
    _queue(conn, vid)

    pipeline.run(conn, cfg)

    updated = store.get_video(conn, vid)
    assert (updated.language, updated.source) == ("en", "human_subs")
    assert FakePool.instances[0].transcribed == []
    assert not (cfg.paths.audio_dir / f"{vid}.webm").exists()  # the detection audio is discarded


def test_run_with_no_auto_videos_never_samples_or_downloads_extra(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    downloads = patch_fetch(monkeypatch, tmp_path)
    _queue(conn, "aaaaaaaaaaa", language="bn")

    result = pipeline.run(conn, cfg)

    assert result.detected == 0
    assert FakePool.instances[0].sampled == []
    assert downloads == ["aaaaaaaaaaa"]


def test_auto_threshold_comes_from_config(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    patch_fetch(monkeypatch, tmp_path)
    _samples(monkeypatch, **{vid: BANGLISH_SAMPLE})  # ~30% Bangla function words
    _queue(conn, vid)
    strict = dataclasses.replace(cfg, asr=dataclasses.replace(cfg.asr, auto_english_below=0.9))

    pipeline.run(conn, strict)

    assert store.get_video(conn, vid).language == "en"  # 30% is below 90%


# --- Facebook ---------------------------------------------------------------------------

FB_ID = "fb_123456789012345"
FB_URL = "https://www.facebook.com/watch/?v=123456789012345"


def test_add_targets_queues_facebook_videos_with_a_prefixed_id_and_platform(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    urls = [video_url("aaaaaaaaaaa"), FB_URL]
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: urls)

    new_ids = pipeline.add_targets(conn, cfg, ["some target"])

    assert new_ids == ["aaaaaaaaaaa", FB_ID]
    assert {v.id: v.platform for v in store.list_videos(conn)} == {
        "aaaaaaaaaaa": "youtube", FB_ID: "facebook",
    }
    assert store.get_video(conn, FB_ID).url == FB_URL


def test_add_targets_skips_a_known_facebook_video(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    store.upsert_video(conn, store.Video(id=FB_ID, url=FB_URL, platform="facebook", status="transcribed"))
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [FB_URL])

    assert pipeline.add_targets(conn, cfg, ["t"]) == []


def test_add_targets_resolves_an_opaque_facebook_link(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    short = "https://fb.watch/abcDEF12/"
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [short])
    monkeypatch.setattr(
        pipeline.fetch, "get_video_info",
        lambda url, cfg: fetch.VideoInfo(
            id=FB_ID, url=FB_URL, title=None, channel=None, channel_id=None,
            upload_date=None, duration=None, description=None, platform="facebook",
        ),
    )

    assert pipeline.add_targets(conn, cfg, ["t"]) == [FB_ID]
    video = store.get_video(conn, FB_ID)
    assert (video.url, video.platform) == (FB_URL, "facebook")  # the canonical URL, not the short link


def test_add_targets_skips_an_opaque_link_that_cannot_be_resolved(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    short = "https://fb.watch/gone/"
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: [short])

    def login_wall(url, cfg):
        raise RuntimeError("Cannot parse data: login required")

    monkeypatch.setattr(pipeline.fetch, "get_video_info", login_wall)

    assert pipeline.add_targets(conn, cfg, ["t"]) == []
    assert store.list_videos(conn) == []
    assert short in caplog.text
    assert "login required" in caplog.text  # the real reason, not a generic message
    assert "cookies_from_browser" in caplog.text  # and what to do about it


def test_add_targets_skips_unsupported_sites_instead_of_queueing_the_raw_url(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(pipeline.fetch, "expand_to_video_urls", lambda target, cfg: ["https://vimeo.com/1"])

    assert pipeline.add_targets(conn, cfg, ["t"]) == []
    assert store.list_videos(conn) == []
    assert "https://vimeo.com/1" in caplog.text


def test_transcribe_one_facebook_video_uses_asr_and_leaves_a_single_row(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_RESULTS[FB_ID] = [store.Segment(idx=0, start_s=0.0, end_s=3.0, text="ফেসবুক")]
    store.upsert_video(conn, store.Video(id=FB_ID, url=FB_URL, platform="facebook", status="queued"))
    video = store.get_video(conn, FB_ID)

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), video)

    assert [v.id for v in store.list_videos(conn)] == [FB_ID]
    updated = store.get_video(conn, FB_ID)
    assert (updated.status, updated.source, updated.platform) == ("transcribed", "asr", "facebook")
    assert [s.text for s in store.get_segments(conn, FB_ID)] == ["ফেসবুক"]
    assert (cfg.paths.videos_dir / FB_ID / "transcript.txt").exists()
    assert f"- id: {FB_ID}" in ledger.ledger_path(cfg).read_text(encoding="utf-8")


def test_transcribe_one_stores_under_the_queued_row_even_if_the_extractor_id_differs(conn, tmp_path, monkeypatch):
    """The queued row is the video's identity. The old code upserted under
    whatever id yt-dlp reported, leaving the queued row stuck beside a duplicate."""
    cfg = make_config(tmp_path)
    patch_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(
        pipeline.fetch, "get_video_info",
        lambda url, cfg: fetch.VideoInfo(
            id="someOtherId", url=url, title="T", channel="C", channel_id="c",
            upload_date=None, duration=60.0, description=None,
        ),
    )
    _TRANSCRIBE_RESULTS["aaaaaaaaaaa"] = [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="x")]
    store.upsert_video(conn, store.Video(id="aaaaaaaaaaa", url=video_url("aaaaaaaaaaa"), status="queued"))

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, "aaaaaaaaaaa"))

    assert [(v.id, v.status) for v in store.list_videos(conn)] == [("aaaaaaaaaaa", "transcribed")]


# --- reexport --------------------------------------------------------------------------

def _transcribed(conn, vid, text="hello", **video_fields):
    store.upsert_video(conn, store.Video(
        id=vid, url=video_url(vid), title=f"Title {vid}", channel="Chan", duration=60.0, language="bn",
        **video_fields,
    ))
    store.replace_segments(conn, vid, [store.Segment(idx=0, start_s=0.0, end_s=60.0, text=text)])
    store.set_video_source(conn, vid, "asr", "m")
    store.set_video_status(conn, vid, "transcribed")


def test_reexport_rebuilds_every_transcribed_video_without_asr(conn, tmp_path):
    cfg = make_config(tmp_path)
    _transcribed(conn, "aaaaaaaaaaa")
    _transcribed(conn, "bbbbbbbbbbb")
    store.upsert_video(conn, store.Video(id="qqqqqqqqqqq", url=video_url("qqqqqqqqqqq")))  # still queued

    done = pipeline.reexport(conn, cfg)

    assert sorted(done) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    shared = cfg.paths.transcribed_files_dir
    folder = shared / date.today().isoformat() / "Chan - Title aaaaaaaaaaa"
    assert (folder / "metadata.json").exists() and (folder / "transcript.vtt").exists()
    assert (cfg.paths.videos_dir / "aaaaaaaaaaa" / "transcript.json").exists()
    assert not (cfg.paths.videos_dir / "qqqqqqqqqqq").exists()
    index = (shared / "INDEX.md").read_text(encoding="utf-8")
    assert "aaaaaaaaaaa" in index and "bbbbbbbbbbb" in index and "qqqqqqqqqqq" not in index


def test_reexport_limits_to_the_given_ids_and_skips_unknown_ones(conn, tmp_path):
    cfg = make_config(tmp_path)
    _transcribed(conn, "aaaaaaaaaaa")
    _transcribed(conn, "bbbbbbbbbbb")

    assert pipeline.reexport(conn, cfg, ["bbbbbbbbbbb", "nope"]) == ["bbbbbbbbbbb"]
    assert not (cfg.paths.videos_dir / "aaaaaaaaaaa").exists()


def test_reexport_converts_an_id_named_folder_and_applies_the_glossary(conn, tmp_path):
    cfg = make_config(tmp_path)
    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text("TrustEd Ireland <= Trusted Ireland\n", encoding="utf-8")
    cfg = dataclasses.replace(cfg, asr=dataclasses.replace(cfg.asr, glossary=glossary_file))
    _transcribed(conn, "aaaaaaaaaaa", text="apply to Trusted Ireland")
    legacy = cfg.paths.transcribed_files_dir / "2026-09-24" / "aaaaaaaaaaa"  # the id-named layout
    legacy.mkdir(parents=True)

    pipeline.reexport(conn, cfg)

    folder = cfg.paths.transcribed_files_dir / "2026-09-24" / "Chan - Title aaaaaaaaaaa"
    assert not legacy.exists()
    assert (folder / "transcript.txt").read_text(encoding="utf-8") == "apply to Trusted Ireland\n"
    assert (folder / "transcript.clean.txt").read_text(encoding="utf-8") == "apply to TrustEd Ireland\n"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["glossary_replacements"] == 1
    assert store.get_segments(conn, "aaaaaaaaaaa")[0].text == "apply to Trusted Ireland"  # DB stays raw


def test_transcribe_one_records_quality_and_confidence_in_the_shared_copy(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "ffffffffffff"[:11]
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_RESULTS[vid] = [
        store.Segment(idx=0, start_s=0.0, end_s=60.0, text=" ".join(["শব্দ"] * 120), avg_logprob=-1.5)
    ]
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    pipeline.transcribe_one(conn, cfg, FakePool(cfg), store.get_video(conn, vid))

    folder = cfg.paths.transcribed_files_dir / date.today().isoformat() / f"Chan - Title {vid}"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["quality"]["low_confidence"] == [0]
    assert meta["quality"]["words"] == 120
    assert "low-confidence" in " ".join(meta["quality"]["flags"])
    seg = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))["segments"][0]
    assert seg["avg_logprob"] == -1.5
