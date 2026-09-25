from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bytx import fetch, ledger, pipeline, store
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
            models_dir=tmp_path / "models",
            transcribed_files_dir=tmp_path / "yt-transcribed-files",
        ),
        asr=AsrConfig(
            model="m", device="auto", compute_type="float16", beam_size=1,
            vad_filter=False, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
            repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="",
            models={"m": AsrModelConfig(key="m", source="org/m", language="bn")},
        ),
        fetch=FetchConfig(
            force_asr=False, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""
        ),
        ollama=OllamaConfig(host="http://x", embed_model="bge-m3", chat_model="gemma3:4b", num_ctx=8192),
        rag=RagConfig(chunk_seconds=75, chunk_overlap_seconds=15, top_k=8),
        summarize=SummarizeConfig(enabled=summarize_enabled, map_window_minutes=10, language="bn"),
    )


class FakeTranscriber:
    """Replaces bytx.pipeline.Transcriber. Records construction and calls;
    per-video results/errors are looked up by audio path stem (~video id).
    """

    instances: list[FakeTranscriber] = []

    def __init__(self, model_dir, model_cfg, asr_cfg):
        self.model_dir = model_dir
        self.unloaded = False
        self.calls: list[Path] = []
        FakeTranscriber.instances.append(self)

    def transcribe(self, audio_path: Path) -> TranscribeResult:
        self.calls.append(audio_path)
        if audio_path.stem in _TRANSCRIBE_FAILURES:
            raise RuntimeError(_TRANSCRIBE_FAILURES[audio_path.stem])
        segs = _TRANSCRIBE_RESULTS.get(
            audio_path.stem, [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="default text")]
        )
        return TranscribeResult(
            segments=segs, device="cuda", compute_type="float16", language="bn",
            stats=_TRANSCRIBE_STATS.get(audio_path.stem, TranscribeStats(speech_seconds=60.0)),
        )

    def unload(self) -> None:
        self.unloaded = True


_TRANSCRIBE_RESULTS: dict[str, list[store.Segment]] = {}
_TRANSCRIBE_FAILURES: dict[str, str] = {}
_TRANSCRIBE_STATS: dict[str, TranscribeStats] = {}


class FakeOllamaClient:
    """Replaces bytx.pipeline.OllamaClient as a context manager."""

    instances: list[FakeOllamaClient] = []

    def __init__(self, ollama_cfg):
        self.cfg = ollama_cfg
        self.closed = False
        FakeOllamaClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def embed(self, texts, model=None):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def chat(self, messages, model=None, num_ctx=None, temperature=None):
        return "সারাংশ পাঠ্য"


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch):
    FakeTranscriber.instances = []
    FakeOllamaClient.instances = []
    _TRANSCRIBE_RESULTS.clear()
    _TRANSCRIBE_FAILURES.clear()
    _TRANSCRIBE_STATS.clear()
    monkeypatch.setattr(pipeline, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(pipeline, "OllamaClient", FakeOllamaClient)
    yield


def patch_fetch(monkeypatch, tmp_path, *, human_subs: dict[str, list[store.Segment]] | None = None,
                 info_failures: set[str] | None = None):
    """Wires fetch.get_video_info / download_audio / fetch_human_subtitle_segments
    to deterministic fakes based on the video id extracted from the URL.
    """
    human_subs = human_subs or {}
    info_failures = info_failures or set()

    def fake_get_video_info(url, cfg):
        vid = fetch.extract_video_id(url)
        if vid in info_failures:
            raise RuntimeError(f"metadata fetch failed for {vid}")
        return fetch.VideoInfo(
            id=vid, url=url, title=f"Title {vid}", channel="Chan", channel_id="c1",
            upload_date="20260101", duration=60.0, description="desc",
            human_subtitle_url=(f"http://subs/{vid}.vtt" if vid in human_subs else None),
            human_subtitle_lang=("bn" if vid in human_subs else None),
        )

    def fake_download_audio(url, out_dir, cfg):
        vid = fetch.extract_video_id(url)
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
        pipeline.transcribe_one(conn, cfg, FakeTranscriber(None, None, None), store.get_video(conn, vid))

    assert "token limit" in caplog.text
    assert store.get_video(conn, vid).status == "transcribed"  # a warning, not a failure


def test_transcribe_one_warns_when_little_speech_detected(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "ccccccccccc"
    patch_fetch(monkeypatch, tmp_path)  # fake video duration is 60s
    _TRANSCRIBE_STATS[vid] = TranscribeStats(speech_seconds=30.0)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakeTranscriber(None, None, None), store.get_video(conn, vid))

    assert "detected as speech" in caplog.text


def test_transcribe_one_no_warning_for_healthy_asr(conn, tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    vid = "ddddddddddd"
    patch_fetch(monkeypatch, tmp_path)
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))

    with caplog.at_level("WARNING", logger="bytx.pipeline"):
        pipeline.transcribe_one(conn, cfg, FakeTranscriber(None, None, None), store.get_video(conn, vid))

    assert caplog.text == ""


def test_transcribe_one_prefers_human_subs(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "aaaaaaaaaaa"
    subs = [store.Segment(idx=0, start_s=0.0, end_s=2.0, text="মানুষের সাবটাইটেল")]
    patch_fetch(monkeypatch, tmp_path, human_subs={vid: subs})
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)

    transcriber = FakeTranscriber(None, None, None)
    pipeline.transcribe_one(conn, cfg, transcriber, video)

    assert transcriber.calls == []  # ASR must not run when human subs exist
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

    transcriber = FakeTranscriber(None, None, None)
    pipeline.transcribe_one(conn, cfg, transcriber, video)

    out_dir = cfg.paths.transcribed_files_dir / f"Chan - Title {vid}"
    transcript = out_dir / "transcript.txt"
    assert transcript.exists()
    assert transcript.read_text(encoding="utf-8") == "মানুষের সাবটাইটেল\n"

    ledger_content = ledger.ledger_path(cfg).read_text(encoding="utf-8")
    assert f"- id: {vid}" in ledger_content


def test_transcribe_one_uses_asr_and_deletes_audio(conn, tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    vid = "bbbbbbbbbbb"
    patch_fetch(monkeypatch, tmp_path)
    _TRANSCRIBE_RESULTS[vid] = [store.Segment(idx=0, start_s=0.0, end_s=3.0, text="এএসআর ফলাফল")]
    store.upsert_video(conn, store.Video(id=vid, url=video_url(vid), status="queued"))
    video = store.get_video(conn, vid)

    transcriber = FakeTranscriber(None, None, None)
    pipeline.transcribe_one(conn, cfg, transcriber, video)

    assert len(transcriber.calls) == 1
    audio_path = transcriber.calls[0]
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
    transcriber = FakeTranscriber(None, None, None)

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.transcribe_one(conn, cfg, transcriber, video)

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
    assert FakeTranscriber.instances[0].unloaded is True


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

    assert FakeTranscriber.instances == []  # never constructed: no queued work
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

    pipeline.transcribe_one(conn, cfg, FakeTranscriber(None, None, None), store.get_video(conn, vid))

    assert store.get_video(conn, vid).status == "transcribed"
