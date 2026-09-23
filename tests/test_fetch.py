from __future__ import annotations

from pathlib import Path

import pytest

from bytx import fetch
from bytx.config import (
    AsrConfig,
    Config,
    FetchConfig,
    OllamaConfig,
    PathsConfig,
    RagConfig,
    SummarizeConfig,
)


def make_config(**fetch_overrides) -> Config:
    fetch_defaults = dict(
        force_asr=False, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""
    )
    fetch_defaults.update(fetch_overrides)
    return Config(
        root=Path("."),
        paths=PathsConfig(data_dir=Path("data"), models_dir=Path("models")),
        asr=AsrConfig(
            model="x", device="auto", compute_type="float16", beam_size=5,
            vad_filter=True, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
            repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="", models={},
        ),
        fetch=FetchConfig(**fetch_defaults),
        ollama=OllamaConfig(host="http://x", embed_model="e", chat_model="c", num_ctx=8192),
        rag=RagConfig(chunk_seconds=75, chunk_overlap_seconds=15, top_k=8),
        summarize=SummarizeConfig(enabled=True, map_window_minutes=10, language="bn"),
    )


# --- normalize_channel_url ---------------------------------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/@SomeChannel", "https://www.youtube.com/@SomeChannel/videos"),
        ("https://www.youtube.com/@SomeChannel/", "https://www.youtube.com/@SomeChannel/videos"),
        ("https://www.youtube.com/channel/UC123", "https://www.youtube.com/channel/UC123/videos"),
        ("https://www.youtube.com/c/SomeName", "https://www.youtube.com/c/SomeName/videos"),
        ("https://www.youtube.com/user/SomeUser", "https://www.youtube.com/user/SomeUser/videos"),
        # Already has a tab or is a single video / playlist: unchanged
        ("https://www.youtube.com/@SomeChannel/videos", "https://www.youtube.com/@SomeChannel/videos"),
        ("https://www.youtube.com/@SomeChannel/shorts", "https://www.youtube.com/@SomeChannel/shorts"),
        ("https://www.youtube.com/watch?v=abc123", "https://www.youtube.com/watch?v=abc123"),
        ("https://www.youtube.com/playlist?list=PL123", "https://www.youtube.com/playlist?list=PL123"),
    ],
)
def test_normalize_channel_url(url, expected):
    assert fetch.normalize_channel_url(url) == expected


# --- expand_to_video_urls (yt_dlp mocked) ------------------------------------

class _FakeYoutubeDL:
    """Records the URL it was asked to extract and returns a canned info dict."""

    calls: list[str] = []
    responses: dict[str, dict | None] = {}

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        _FakeYoutubeDL.calls.append(url)
        return _FakeYoutubeDL.responses.get(url)


@pytest.fixture(autouse=True)
def _patch_yt_dlp(monkeypatch):
    _FakeYoutubeDL.calls = []
    _FakeYoutubeDL.responses = {}
    monkeypatch.setattr(fetch.yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    yield


def test_expand_single_video(monkeypatch):
    url = "https://www.youtube.com/watch?v=abc123"
    _FakeYoutubeDL.responses[url] = {"webpage_url": url}
    result = fetch.expand_to_video_urls(url, make_config())
    assert result == [url]
    assert _FakeYoutubeDL.calls == [url]


def test_expand_playlist(monkeypatch):
    url = "https://www.youtube.com/playlist?list=PL123"
    _FakeYoutubeDL.responses[url] = {
        "entries": [
            {"id": "v1", "url": "v1"},
            {"id": "v2", "url": "https://www.youtube.com/watch?v=v2"},
        ]
    }
    result = fetch.expand_to_video_urls(url, make_config())
    assert result == [
        "https://www.youtube.com/watch?v=v1",
        "https://www.youtube.com/watch?v=v2",
    ]


def test_expand_channel_normalizes_and_uses_videos_tab():
    url = "https://www.youtube.com/@SomeChannel"
    normalized = "https://www.youtube.com/@SomeChannel/videos"
    _FakeYoutubeDL.responses[normalized] = {
        "entries": [{"id": "v1", "url": "v1"}]
    }
    result = fetch.expand_to_video_urls(url, make_config())
    assert result == ["https://www.youtube.com/watch?v=v1"]
    assert _FakeYoutubeDL.calls == [normalized]


def test_expand_returns_empty_on_none_info():
    url = "https://www.youtube.com/watch?v=deleted"
    _FakeYoutubeDL.responses[url] = None
    assert fetch.expand_to_video_urls(url, make_config()) == []


def test_expand_nested_entries_flattened():
    url = "https://www.youtube.com/@SomeChannel/videos"
    _FakeYoutubeDL.responses[url] = {
        "entries": [
            {"entries": [{"id": "v1", "url": "v1"}, {"id": "v2", "url": "v2"}]},
            {"id": "v3", "url": "v3"},
        ]
    }
    result = fetch.expand_to_video_urls(url, make_config())
    assert result == [
        "https://www.youtube.com/watch?v=v1",
        "https://www.youtube.com/watch?v=v2",
        "https://www.youtube.com/watch?v=v3",
    ]


def test_expand_from_text_file(tmp_path):
    v1 = "https://www.youtube.com/watch?v=v1"
    v2 = "https://www.youtube.com/watch?v=v2"
    _FakeYoutubeDL.responses[v1] = {"webpage_url": v1}
    _FakeYoutubeDL.responses[v2] = {"webpage_url": v2}

    list_file = tmp_path / "urls.txt"
    list_file.write_text(f"# comment\n\n{v1}\n{v2}\n", encoding="utf-8")

    result = fetch.expand_to_video_urls(str(list_file), make_config())
    assert result == [v1, v2]


def test_get_video_info_picks_human_bn_subtitle():
    url = "https://www.youtube.com/watch?v=abc123"
    _FakeYoutubeDL.responses[url] = {
        "id": "abc123",
        "webpage_url": url,
        "title": "শিরোনাম",
        "channel": "Test Channel",
        "channel_id": "c1",
        "upload_date": "20260101",
        "duration": 300.0,
        "description": "desc",
        "subtitles": {
            "bn": [
                {"ext": "srv3", "url": "http://x/bn.srv3"},
                {"ext": "vtt", "url": "http://x/bn.vtt"},
            ],
            "en": [{"ext": "vtt", "url": "http://x/en.vtt"}],
        },
        "automatic_captions": {
            "bn": [{"ext": "vtt", "url": "http://x/auto-bn.vtt"}],
        },
    }
    info = fetch.get_video_info(url, make_config())
    assert info.id == "abc123"
    assert info.title == "শিরোনাম"
    assert info.human_subtitle_url == "http://x/bn.vtt"
    assert info.human_subtitle_lang == "bn"


def test_get_video_info_no_human_subtitle_available():
    url = "https://www.youtube.com/watch?v=abc123"
    _FakeYoutubeDL.responses[url] = {
        "id": "abc123",
        "webpage_url": url,
        "title": "t",
        "subtitles": {},
        "automatic_captions": {"bn": [{"ext": "vtt", "url": "http://x/auto-bn.vtt"}]},
    }
    info = fetch.get_video_info(url, make_config())
    assert info.human_subtitle_url is None
    assert info.human_subtitle_lang is None


# --- parse_vtt ----------------------------------------------------------------

def test_parse_vtt_basic_cues():
    text = """WEBVTT

00:00:00.000 --> 00:00:02.500
Hello

00:00:02.500 --> 00:00:05.000 align:start position:0%
World
Second line
"""
    segs = fetch.parse_vtt(text)
    assert len(segs) == 2
    assert segs[0].start_s == 0.0
    assert segs[0].end_s == 2.5
    assert segs[0].text == "Hello"
    assert segs[1].start_s == 2.5
    assert segs[1].text == "World Second line"
    assert segs[0].idx == 0 and segs[1].idx == 1


def test_parse_vtt_strips_tags_and_skips_empty_cues():
    text = """WEBVTT

00:00:00.000 --> 00:00:01.000
<b>বাংলা</b> <00:00:00.500>টেক্সট

00:00:01.000 --> 00:00:02.000

"""
    segs = fetch.parse_vtt(text)
    assert len(segs) == 1
    assert segs[0].text == "বাংলা টেক্সট"


def test_parse_vtt_empty_text():
    assert fetch.parse_vtt("") == []
    assert fetch.parse_vtt("WEBVTT\n\n") == []
