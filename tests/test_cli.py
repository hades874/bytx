from __future__ import annotations

import pytest
from typer.testing import CliRunner

from bytx import cli, fetch, store

runner = CliRunner()

_CONFIG = """
[asr]
model = "bn-model"
[asr.language_models]
bn = "bn-model"
en = "multi"
[asr.models.bn-model]
source = "org/bn"
language = "bn"
[asr.models.multi]
source = "org/multi"
language = ""
already_ct2 = true
"""

URL = "https://www.youtube.com/watch?v=aaaaaaaaaaa"


@pytest.fixture()
def config_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(_CONFIG, encoding="utf-8")
    return str(path)


def db(tmp_path):
    return store.connect(tmp_path / "data" / "library.db")


@pytest.fixture()
def one_url(monkeypatch):
    monkeypatch.setattr(fetch, "expand_to_video_urls", lambda target, cfg: [URL])


# --- add --lang --------------------------------------------------------------------

def test_add_defaults_to_the_configured_language(config_path, tmp_path, one_url):
    result = runner.invoke(cli.app, ["add", URL, "--config", config_path])
    assert result.exit_code == 0, result.output
    assert db(tmp_path).execute("SELECT language FROM videos").fetchone()[0] == "bn"


@pytest.mark.parametrize(("flag", "stored"), [("en", "en"), ("bn", "bn"), ("EN", "en"), ("English", "en")])
def test_add_lang_is_stored_on_the_queued_video(config_path, tmp_path, one_url, flag, stored):
    result = runner.invoke(cli.app, ["add", URL, "--lang", flag, "--config", config_path])
    assert result.exit_code == 0, result.output
    row = db(tmp_path).execute("SELECT language, status FROM videos").fetchone()
    assert (row["language"], row["status"]) == (stored, "queued")


@pytest.mark.parametrize("flag", ["auto", "AUTO"])
def test_add_auto_queues_the_video_with_no_language_yet(config_path, tmp_path, one_url, flag):
    result = runner.invoke(cli.app, ["add", URL, "--lang", flag, "--config", config_path])
    assert result.exit_code == 0, result.output
    row = db(tmp_path).execute("SELECT language, status FROM videos").fetchone()
    assert (row["language"], row["status"]) == (None, "queued")  # run decides it


def test_auto_is_only_for_add_not_for_filtering_or_benchmarking(config_path):
    for args in (["list", "--lang", "auto"], ["bench", URL, "--lang", "auto"]):
        result = runner.invoke(cli.app, [*args, "--config", config_path])
        assert result.exit_code == 1, args
        assert "Unsupported language 'auto'" in result.output


def test_unsupported_language_error_lists_auto_only_where_it_is_accepted(config_path, one_url):
    add = runner.invoke(cli.app, ["add", URL, "--lang", "fr", "--config", config_path])
    listing = runner.invoke(cli.app, ["list", "--lang", "fr", "--config", config_path])
    assert "auto, bn, en" in add.output
    assert "auto" not in listing.output


def test_add_rejects_an_unsupported_language_before_touching_anything(config_path, tmp_path, one_url):
    result = runner.invoke(cli.app, ["add", URL, "--lang", "fr", "--config", config_path])
    assert result.exit_code == 1
    assert "Unsupported language 'fr'" in result.output
    assert "bn, en" in result.output
    assert not (tmp_path / "data").exists()  # rejected before the config was even loaded


def test_add_force_asr_no_longer_crashes_and_reaches_the_run(config_path, one_url, monkeypatch):
    """`cfg.fetch.force_asr = True` used to raise FrozenInstanceError."""
    seen = []
    monkeypatch.setattr(cli, "_do_run", lambda conn, cfg, limit, summarize_enabled=None: seen.append(cfg))

    result = runner.invoke(cli.app, ["add", URL, "--force-asr", "--run", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert seen[0].fetch.force_asr is True


def test_add_without_force_asr_keeps_the_config_value(config_path, one_url, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_do_run", lambda conn, cfg, limit, summarize_enabled=None: seen.append(cfg))

    runner.invoke(cli.app, ["add", URL, "--run", "--config", config_path])

    assert seen[0].fetch.force_asr is False


# --- list / show ---------------------------------------------------------------------

def _seed(tmp_path):
    conn = db(tmp_path)
    store.upsert_video(conn, store.Video(id="en1", url="u1", title="T-en", language="en", status="transcribed"))
    store.upsert_video(conn, store.Video(id="bn1", url="u2", title="T-bn", language="bn", status="transcribed"))
    store.upsert_video(conn, store.Video(id="new1", url="u3", title="T-new", status="queued"))
    conn.close()


def test_list_shows_a_language_column(config_path, tmp_path):
    _seed(tmp_path)
    result = runner.invoke(cli.app, ["list", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "Lang" in result.output
    assert "en1" in result.output and "bn1" in result.output


def test_list_filters_by_language(config_path, tmp_path):
    _seed(tmp_path)
    result = runner.invoke(cli.app, ["list", "--lang", "en", "--config", config_path])
    assert "en1" in result.output
    assert "bn1" not in result.output
    assert "new1" not in result.output


def test_list_rejects_an_unsupported_language(config_path):
    result = runner.invoke(cli.app, ["list", "--lang", "xx", "--config", config_path])
    assert result.exit_code == 1
    assert "Unsupported language" in result.output


def test_show_prints_the_language_in_brackets(config_path, tmp_path):
    _seed(tmp_path)
    result = runner.invoke(cli.app, ["show", "en1", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "[en]" in result.output  # not swallowed as Rich markup


# --- config errors keep their bracketed text -------------------------------------------

def test_config_errors_are_not_eaten_by_rich_markup(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(_CONFIG.replace('en = "multi"', 'fr = "multi"'), encoding="utf-8")
    result = runner.invoke(cli.app, ["list", "--config", str(bad)])
    assert result.exit_code == 1
    assert "[asr.language_models]" in result.output
    assert "unsupported language 'fr'" in result.output


# --- doctor / setup cover every configured model ----------------------------------------

class _NoOllama:
    def __init__(self, cfg):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def is_reachable(self):
        return False

    def close(self):
        pass


class _RunningOllama(_NoOllama):
    def is_reachable(self):
        return True

    def has_model(self, model):
        return True


class _Done:
    returncode = 0
    stdout = "2026.01.01"


def test_doctor_checks_each_configured_asr_model(config_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Done())

    result = runner.invoke(cli.app, ["doctor", "--config", config_path])

    assert result.exit_code == 1  # models are not downloaded in the test dir
    assert "ASR model 'bn-model' (org/bn) is not downloaded" in result.output
    assert "ASR model 'multi' (org/multi) is not downloaded" in result.output


def test_setup_downloads_every_configured_model_then_pulls_ollama_models(config_path, tmp_path, monkeypatch):
    downloaded, calls = [], []
    monkeypatch.setattr(cli, "OllamaClient", _RunningOllama)
    monkeypatch.setattr(cli.models, "download", lambda m, force=False: downloaded.append(m.key) or tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or _Done())

    result = runner.invoke(cli.app, ["setup", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert downloaded == ["bn-model", "multi"]
    pulled = [c[2] for c in calls if c[:2] == ["ollama", "pull"]]
    assert pulled == ["bge-m3", "gemma3:4b"]


def test_setup_stops_when_a_model_fails_to_download(config_path, monkeypatch):
    def fail(model_cfg, force=False):
        raise OSError("connection reset")

    calls = []
    monkeypatch.setattr(cli.models, "download", fail)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or _Done())

    result = runner.invoke(cli.app, ["setup", "--config", config_path])

    assert result.exit_code == 1
    assert "connection reset" in result.output
    assert calls == []  # did not go on to Ollama


def test_setup_explains_that_an_unconverted_model_needs_the_convert_script(config_path):
    result = runner.invoke(cli.app, ["setup", "--config", config_path])  # bn-model is not already_ct2
    assert result.exit_code == 1
    assert "scripts/convert_model.py" in result.output


# --- search output ----------------------------------------------------------------------

@pytest.mark.parametrize(("text", "query"), [
    ("Apple is so innovative year after year", "innovative"),
    ("আপনারা যারা বাংলাদেশে ব্যাচেলর কম্পিট করেছেন", "বাংলাদেশ"),
])
def test_keyword_search_shows_the_matched_word(config_path, tmp_path, text, query):
    """FTS marks the match as [word]; Rich used to eat that as a style tag and
    print an empty snippet."""
    conn = db(tmp_path)
    store.upsert_video(conn, store.Video(id="v1", url="u", title="T", language="en", status="transcribed"))
    store.replace_segments(conn, "v1", [store.Segment(idx=0, start_s=0, end_s=5, text=text)])
    conn.close()

    result = runner.invoke(cli.app, ["search", query, "--config", config_path])

    assert result.exit_code == 0, result.output
    assert f"[{query}" in result.output


# --- Facebook / platform-aware output ---------------------------------------------------

FB_URL = "https://www.facebook.com/watch/?v=123456789012345"


def _seed_library(tmp_path):
    conn = db(tmp_path)
    for vid, url, platform in (
        ("aaaaaaaaaaa", URL, "youtube"),
        ("fb_123456789012345", FB_URL, "facebook"),
    ):
        store.upsert_video(
            conn, store.Video(id=vid, url=url, title=f"T {vid}", channel="Chan", platform=platform, status="indexed")
        )
        store.replace_segments(
            conn, vid, [store.Segment(idx=0, start_s=75.0, end_s=80.0, text="মজার কথা হচ্ছে এখানে")]
        )
    return conn


def test_add_queues_a_facebook_url_with_its_platform(config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "expand_to_video_urls", lambda target, cfg: [FB_URL])
    result = runner.invoke(cli.app, ["add", FB_URL, "--config", config_path])
    assert result.exit_code == 0, result.output
    row = db(tmp_path).execute("SELECT id, platform, status FROM videos").fetchone()
    assert (row["id"], row["platform"], row["status"]) == ("fb_123456789012345", "facebook", "queued")


def test_search_links_each_hit_to_its_own_platform(config_path, tmp_path):
    _seed_library(tmp_path)
    result = runner.invoke(cli.app, ["search", "মজার", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "https://youtu.be/aaaaaaaaaaa?t=75" in result.output
    assert "https://www.facebook.com/watch/?v=123456789012345&t=75" in result.output


def test_list_shows_the_platform_and_can_filter_by_it(config_path, tmp_path):
    _seed_library(tmp_path)

    everything = runner.invoke(cli.app, ["list", "--config", config_path])
    facebook_only = runner.invoke(cli.app, ["list", "--platform", "facebook", "--config", config_path])

    assert "aaaaaaaaaaa" in everything.output and "fb_123456789012345" in everything.output
    assert "facebook" in facebook_only.output
    assert "fb_123456789012345" in facebook_only.output and "aaaaaaaaaaa" not in facebook_only.output


def test_list_rejects_an_unknown_platform(config_path):
    result = runner.invoke(cli.app, ["list", "--platform", "vimeo", "--config", config_path])
    assert result.exit_code == 1
    assert "Unknown platform 'vimeo'" in result.output


@pytest.mark.parametrize("flag", ["facebook", "Facebook", " FACEBOOK "])
def test_list_platform_filter_ignores_case(config_path, tmp_path, flag):
    _seed_library(tmp_path)
    result = runner.invoke(cli.app, ["list", "--platform", flag, "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "fb_123456789012345" in result.output and "aaaaaaaaaaa" not in result.output


def test_doctor_reports_a_missing_deno_and_never_checks_ffmpeg(config_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli, "_deno_path", lambda: None)
    result = runner.invoke(cli.app, ["doctor", "--config", config_path])
    assert result.exit_code == 1
    assert "which YouTube downloads need" in result.output and "Facebook works without it" in result.output
    assert "ffmpeg" not in result.output


def test_doctor_finds_the_deno_that_ships_with_bytx():
    assert cli._deno_path() is not None  # yt-dlp[deno] installs it next to python


class _NvidiaSmi:
    returncode = 0
    stdout = "GPU 0: NVIDIA GeForce RTX 4070 (UUID: GPU-x)\n"


def test_doctor_suggests_the_gpu_version_for_an_unused_nvidia_gpu(config_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli, "_gpu_extra_installed", lambda: False)
    monkeypatch.setattr(cli.shutil, "which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _NvidiaSmi())
    output = " ".join(runner.invoke(cli.app, ["doctor", "--config", config_path]).output.split())
    assert "RTX 4070" in output and 'uv tool install --reinstall "bytx[gpu]"' in output


def test_doctor_gives_no_gpu_tip_when_the_gpu_version_is_installed(config_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli, "_gpu_extra_installed", lambda: True)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _NvidiaSmi())
    result = runner.invoke(cli.app, ["doctor", "--config", config_path])
    assert "bytx[gpu]" not in result.output


def test_doctor_shows_the_data_folder_and_whether_settings_exist(config_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    with_file = runner.invoke(cli.app, ["doctor", "--config", config_path]).output
    defaults = runner.invoke(cli.app, ["doctor"]).output  # BYTX_HOME has no config.toml
    assert "Data folder:" in with_file and "using the defaults" not in with_file
    assert "using the defaults" in defaults


def test_bench_clip_without_ffmpeg_stops_before_downloading(config_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda tool: None)
    monkeypatch.setattr(fetch, "download_audio", lambda *a: pytest.fail("should not download"))
    result = runner.invoke(cli.app, ["bench", URL, "--clip", "00:00-00:10", "--config", config_path])
    assert result.exit_code == 1
    assert "needs ffmpeg" in result.output


# --- quality / reexport / compare ----------------------------------------------------------

def _transcribed(tmp_path, video_id, text, duration=60.0, language="en", platform="youtube", **seg_fields):
    conn = db(tmp_path)
    store.upsert_video(conn, store.Video(
        id=video_id, url=f"https://www.youtube.com/watch?v={video_id}", title=f"Title {video_id}",
        channel="Chan", duration=duration, language=language, platform=platform,
    ))
    store.replace_segments(
        conn, video_id, [store.Segment(idx=0, start_s=0.0, end_s=duration, text=text, **seg_fields)]
    )
    store.set_video_source(conn, video_id, "asr", "m")
    store.set_video_status(conn, video_id, "transcribed")
    conn.close()


HEALTHY = " ".join(["word"] * 130)  # 130 words/min for a 60 s video


def test_quality_lists_flagged_transcripts_first(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", HEALTHY)
    _transcribed(tmp_path, "thinthinthi", "just three words")

    result = runner.invoke(cli.app, ["quality", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert result.output.index("thinthinthi") < result.output.index("goodgoodgoo")
    assert "1 of 2 transcript(s) flagged" in result.output


def test_quality_flagged_hides_healthy_transcripts(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", HEALTHY)
    _transcribed(tmp_path, "thinthinthi", "just three words")

    result = runner.invoke(cli.app, ["quality", "--flagged", "--config", config_path])

    assert "thinthinthi" in result.output and "goodgoodgoo" not in result.output


def test_quality_for_one_video_explains_the_flags(config_path, tmp_path):
    _transcribed(tmp_path, "thinthinthi", "just three words")

    result = runner.invoke(cli.app, ["quality", "thinthinthi", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert "only 3 words/min" in result.output
    assert "speech may be missing" in result.output
    assert "No per-clip confidence" in result.output


def test_quality_for_an_unknown_video_fails(config_path):
    result = runner.invoke(cli.app, ["quality", "nope", "--config", config_path])
    assert result.exit_code == 1
    assert "No such video" in result.output


def test_reexport_needs_ids_or_all(config_path):
    result = runner.invoke(cli.app, ["reexport", "--config", config_path])
    assert result.exit_code == 1
    assert "--all" in result.output


def test_reexport_all_rewrites_the_shared_folder_and_index(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", HEALTHY)

    result = runner.invoke(cli.app, ["reexport", "--all", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert "Rewrote 1 transcript(s)" in result.output
    shared = tmp_path / "yt-transcribed-files"
    assert (shared / "INDEX.md").exists()
    assert list(shared.glob("*/Chan - Title goodgoodgoo/metadata.json"))


def test_reexport_reports_ids_it_skipped(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", HEALTHY)

    result = runner.invoke(cli.app, ["reexport", "goodgoodgoo", "nope", "--config", config_path])

    assert "Rewrote 1 transcript(s)" in result.output
    assert "nope" in result.output


VTT = """WEBVTT

00:00:01.000 --> 00:00:05.000
this is the reference text

00:00:20.000 --> 00:00:28.000
speech the transcript never captured
"""


def test_compare_against_a_caption_file_writes_a_report(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", "something completely unrelated to it", duration=10.0)
    vtt = tmp_path / "ref.vtt"
    vtt.write_text(VTT, encoding="utf-8")
    report = tmp_path / "report.md"

    result = runner.invoke(cli.app, [
        "compare", "goodgoodgoo", "--against", str(vtt), "--output", str(report), "--config", config_path,
    ])

    assert result.exit_code == 0, result.output
    assert "1 below 50%" in result.output
    text = report.read_text(encoding="utf-8")
    assert "# Comparison: Title goodgoodgoo" in text
    assert "this is the reference text" in text
    assert "speech the transcript never captured" in text  # listed as missing


def test_compare_defaults_to_compare_md_in_the_transcript_folder(config_path, tmp_path):
    _transcribed(tmp_path, "goodgoodgoo", "something completely unrelated to it", duration=10.0)
    runner.invoke(cli.app, ["reexport", "--all", "--config", config_path])
    vtt = tmp_path / "ref.vtt"
    vtt.write_text(VTT, encoding="utf-8")

    result = runner.invoke(cli.app, ["compare", "goodgoodgoo", "--against", str(vtt), "--config", config_path])

    assert result.exit_code == 0, result.output
    (folder,) = (tmp_path / "yt-transcribed-files").glob("*/Chan - Title goodgoodgoo")
    assert (folder / "compare.md").exists()


def test_compare_fetches_youtube_auto_captions(config_path, tmp_path, monkeypatch):
    _transcribed(tmp_path, "goodgoodgoo", "something completely unrelated to it", duration=10.0)
    info = fetch.VideoInfo(
        id="goodgoodgoo", url="u", title="t", channel="c", channel_id="c", upload_date=None,
        duration=10.0, description=None, auto_subtitles={"en": "http://captions/en.json3"},
    )
    monkeypatch.setattr(fetch, "get_video_info", lambda url, cfg: info)
    monkeypatch.setattr(
        fetch, "fetch_auto_caption_segments",
        lambda url: [store.Segment(idx=0, start_s=1, end_s=5, text="the auto caption text")],
    )
    out = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["compare", "goodgoodgoo", "--output", str(out), "--config", config_path])

    assert result.exit_code == 0, result.output
    assert "YouTube auto-captions: the auto caption text" in out.read_text(encoding="utf-8")


def test_compare_explains_when_youtube_has_no_captions_in_the_language(config_path, tmp_path, monkeypatch):
    _transcribed(tmp_path, "goodgoodgoo", "words", duration=10.0)
    info = fetch.VideoInfo(
        id="goodgoodgoo", url="u", title="t", channel="c", channel_id="c", upload_date=None,
        duration=10.0, description=None,
    )
    monkeypatch.setattr(fetch, "get_video_info", lambda url, cfg: info)

    result = runner.invoke(cli.app, ["compare", "goodgoodgoo", "--config", config_path])

    assert result.exit_code == 1
    assert "no English auto-captions" in result.output.replace("\n", " ")


def test_compare_says_when_youtube_hears_another_language(config_path, tmp_path, monkeypatch):
    _transcribed(tmp_path, "goodgoodgoo", "words", duration=10.0)
    info = fetch.VideoInfo(
        id="goodgoodgoo", url="u", title="t", channel="c", channel_id="c", upload_date=None,
        duration=10.0, description=None, auto_spoken_language="hi",
    )
    monkeypatch.setattr(fetch, "get_video_info", lambda url, cfg: info)

    result = runner.invoke(cli.app, ["compare", "goodgoodgoo", "--config", config_path])

    assert result.exit_code == 1
    assert "hears this video as 'hi', not English" in result.output.replace("\n", " ")


def test_compare_rejects_facebook_videos_without_a_caption_file(config_path, tmp_path):
    _transcribed(tmp_path, "fb_123456789012345", "words", duration=10.0, platform="facebook")

    result = runner.invoke(cli.app, ["compare", "fb_123456789012345", "--config", config_path])

    assert result.exit_code == 1
    assert "--against" in result.output


def test_compare_unknown_video_and_missing_file_fail(config_path, tmp_path):
    assert runner.invoke(cli.app, ["compare", "nope", "--config", config_path]).exit_code == 1
    _transcribed(tmp_path, "goodgoodgoo", "words", duration=10.0)
    missing = runner.invoke(cli.app, ["compare", "goodgoodgoo", "--against", "nofile.vtt", "--config", config_path])
    assert missing.exit_code == 1
    assert "No such file" in missing.output


# --- data folder ----------------------------------------------------------------------

def test_the_first_command_creates_the_data_folder(tmp_path, monkeypatch):
    home = tmp_path / "fresh"
    monkeypatch.setenv("BYTX_HOME", str(home))
    result = runner.invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "No videos in the library yet" in result.output
    assert (home / "data" / "library.db").is_file()


def test_run_before_setup_exits_with_a_hint_and_keeps_the_queue(config_path, tmp_path, one_url):
    runner.invoke(cli.app, ["add", URL, "--lang", "bn", "--config", config_path])
    result = runner.invoke(cli.app, ["run", "--config", config_path])  # nothing downloaded in tmp
    assert result.exit_code == 1
    assert "Run `bytx setup`" in result.output
    assert db(tmp_path).execute("SELECT status FROM videos").fetchone()[0] == "queued"


def test_run_prints_why_it_stopped_and_still_succeeds(config_path, monkeypatch):
    stopped = cli.pipeline.RunSummary(transcribed=1, stopped="Ollama (optional) isn't running")
    monkeypatch.setattr(cli.pipeline, "run", lambda *a, **k: stopped)
    result = runner.invoke(cli.app, ["run", "--config", config_path])
    assert result.exit_code == 0
    assert "Ollama (optional) isn't running" in result.output


# --- Ollama is optional ------------------------------------------------------------------

def test_setup_without_ollama_skips_its_models_and_succeeds(config_path, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli.models, "download", lambda m, force=False: tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd) or _Done())

    result = runner.invoke(cli.app, ["setup", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert "Skipped the Q&A and summary models" in result.output
    assert calls == []


@pytest.mark.parametrize("args", [["ask", "why?"], ["search", "x", "--semantic"], ["summarize", "--all"]])
def test_llm_commands_without_ollama_say_how_to_enable_them(config_path, monkeypatch, args):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    result = runner.invoke(cli.app, [*args, "--config", config_path])
    assert result.exit_code == 1
    assert "https://ollama.com" in result.output


def test_keyword_search_works_without_ollama(config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    _transcribed(tmp_path, "aaaaaaaaaaa", "hello world")
    result = runner.invoke(cli.app, ["search", "hello", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "Title aaaaaaaaaaa" in result.output


class _LoadedTranscriber:
    device, compute_type = "cuda", "float16"

    def __init__(self, *args):
        pass

    def unload(self):
        pass


def test_doctor_passes_without_ollama(config_path, tmp_path, monkeypatch):
    import shutil

    from bytx import transcribe

    monkeypatch.setattr(cli, "OllamaClient", _NoOllama)
    monkeypatch.setattr(cli.models, "local_path", lambda m: tmp_path)
    monkeypatch.setattr(transcribe, "Transcriber", _LoadedTranscriber)
    monkeypatch.setattr(shutil, "which", lambda tool: f"/bin/{tool}")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Done())

    result = runner.invoke(cli.app, ["doctor", "--config", config_path])

    assert result.exit_code == 0, result.output
    assert "WARN" in result.output and "Ollama not reachable" in result.output


# --- remove -----------------------------------------------------------------------------

def test_remove_is_a_dry_run_without_yes(config_path, tmp_path):
    _transcribed(tmp_path, "aaaaaaaaaaa", "hello")
    result = runner.invoke(cli.app, ["remove", "aaaaaaaaaaa", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "database row:     yes" in result.output and "Dry run" in result.output
    assert db(tmp_path).execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_remove_with_yes_deletes_the_video(config_path, tmp_path):
    _transcribed(tmp_path, "aaaaaaaaaaa", "hello")
    result = runner.invoke(cli.app, ["remove", "aaaaaaaaaaa", "--yes", "--config", config_path])
    assert result.exit_code == 0, result.output
    assert "Removed aaaaaaaaaaa" in result.output
    assert db(tmp_path).execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 0


@pytest.mark.parametrize(("video_id", "message"), [("zzzzzzzzzzz", "No such video"), ("..", "is not a video id")])
def test_remove_of_an_unknown_or_bad_id_exits_non_zero(config_path, video_id, message):
    result = runner.invoke(cli.app, ["remove", video_id, "--yes", "--config", config_path])
    assert result.exit_code == 1
    assert message in result.output
