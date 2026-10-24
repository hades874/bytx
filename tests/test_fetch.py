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
        paths=PathsConfig(
            data_dir=Path("data"),
            transcribed_files_dir=Path("yt-transcribed-files"),
        ),
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
    # Both supported languages are collected (the caller picks by the video's language),
    # preferring the vtt format; automatic captions never appear.
    assert info.human_subtitles == {"bn": "http://x/bn.vtt", "en": "http://x/en.vtt"}


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
    assert info.human_subtitles == {}


def _info_for(subtitles: dict, **extra) -> fetch.VideoInfo:
    url = "https://www.youtube.com/watch?v=abc123"
    _FakeYoutubeDL.responses[url] = {
        "id": "abc123", "webpage_url": url, "title": "t", "subtitles": subtitles, **extra,
    }
    return fetch.get_video_info(url, make_config())


def test_get_video_info_collects_english_subtitle_variants_first_wins():
    info = _info_for({
        "en-GB": [{"ext": "vtt", "url": "http://x/en-gb.vtt"}],
        "en": [{"ext": "vtt", "url": "http://x/en.vtt"}],
    })
    assert info.human_subtitles == {"en": "http://x/en-gb.vtt"}


def test_get_video_info_ignores_unsupported_subtitle_languages():
    info = _info_for({
        "hi": [{"ext": "vtt", "url": "http://x/hi.vtt"}],
        "fr": [{"ext": "vtt", "url": "http://x/fr.vtt"}],
        "en": [{"ext": "srv3", "url": "http://x/en.srv3"}],  # no vtt: falls back to the first format
    })
    assert info.human_subtitles == {"en": "http://x/en.srv3"}


def test_get_video_info_english_only_subtitles_are_not_mistaken_for_bangla():
    info = _info_for({"en": [{"ext": "vtt", "url": "http://x/en.vtt"}]})
    assert "bn" not in info.human_subtitles


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


# --- Facebook ------------------------------------------------------------------------

FB_URL = "https://www.facebook.com/SomePage/videos/123456789012345"


def _facebook_info(**overrides) -> dict:
    info = {
        "id": "123456789012345",
        "extractor_key": "Facebook",
        "webpage_url": FB_URL,
        "title": "ফেসবুক ভিডিও",
        "uploader": "Some Page",
        "uploader_id": "SomePage",
        "upload_date": "20260101",
        "duration": 120.0,
        "description": "caption line one\nline two",
    }
    info.update(overrides)
    return info


def test_get_video_info_for_a_facebook_video():
    _FakeYoutubeDL.responses[FB_URL] = _facebook_info()
    info = fetch.get_video_info(FB_URL, make_config())
    assert info.id == "fb_123456789012345"
    assert info.platform == "facebook"
    assert info.url == FB_URL
    assert info.title == "ফেসবুক ভিডিও"
    assert info.channel == "Some Page"       # no `channel` key: falls back to the uploader
    assert info.channel_id == "SomePage"     # ... and uploader_id
    assert info.human_subtitles == {}        # nothing to prefer over ASR


def test_facebook_video_without_a_title_is_named_by_its_caption():
    _FakeYoutubeDL.responses[FB_URL] = _facebook_info(title=None)
    assert fetch.get_video_info(FB_URL, make_config()).title == "caption line one"


def test_facebook_video_with_no_title_or_caption_stays_untitled():
    _FakeYoutubeDL.responses[FB_URL] = _facebook_info(title=None, description=None)
    assert fetch.get_video_info(FB_URL, make_config()).title is None


def test_youtube_video_info_keeps_its_bare_id_and_platform():
    info = _info_for({})
    assert (info.id, info.platform) == ("abc123", "youtube")


def test_flat_entries_from_other_extractors_keep_their_own_urls():
    url = "https://www.facebook.com/SomePage/videos"
    _FakeYoutubeDL.responses[url] = {
        "entries": [
            {"id": "1", "ie_key": "Facebook", "url": "https://www.facebook.com/watch/?v=1"},
            {"id": "2", "ie_key": "Facebook"},  # no usable URL: must not become a YouTube link
            {"id": "v3", "ie_key": "Youtube"},
        ]
    }
    assert fetch.expand_to_video_urls(url, make_config()) == [
        "https://www.facebook.com/watch/?v=1",
        "https://www.youtube.com/watch?v=v3",
    ]


def test_expand_a_facebook_video_url_is_left_alone():
    _FakeYoutubeDL.responses[FB_URL] = {"webpage_url": FB_URL}
    assert fetch.expand_to_video_urls(FB_URL, make_config()) == [FB_URL]
    assert _FakeYoutubeDL.calls == [FB_URL]


def test_flat_entries_with_an_unrecognised_extractor_and_no_url_are_dropped():
    url = "https://example.com/list"
    _FakeYoutubeDL.responses[url] = {"entries": [{"id": "9", "ie_key": "Generic"}]}
    assert fetch.expand_to_video_urls(url, make_config()) == []


# --- auto-captions (for `bytx compare`) -----------------------------------------------

def test_auto_captions_take_the_spoken_language_track_not_translations():
    info = _info_for({}, automatic_captions={
        "bn-orig": [{"ext": "vtt", "url": "http://x/bn.vtt"}, {"ext": "json3", "url": "http://x/bn.json3"}],
        "en": [{"ext": "json3", "url": "http://x/en-translated.json3"}],  # machine translation of the Bangla
        "de": [{"ext": "json3", "url": "http://x/de.json3"}],
    })
    assert info.auto_subtitles == {"bn": "http://x/bn.json3"}


def test_auto_captions_without_an_orig_key_fall_back_to_the_plain_codes():
    info = _info_for({}, automatic_captions={
        "en": [{"ext": "json3", "url": "http://x/en.json3"}],
        "fr": [{"ext": "json3", "url": "http://x/fr.json3"}],
    })
    assert info.auto_subtitles == {"en": "http://x/en.json3"}


def test_auto_captions_need_a_json3_format():
    info = _info_for({}, automatic_captions={"en-orig": [{"ext": "vtt", "url": "http://x/en.vtt"}]})
    assert info.auto_subtitles == {}


def test_no_auto_captions_at_all():
    info = _info_for({})
    assert info.auto_subtitles == {} and info.auto_spoken_language is None


def test_the_language_youtube_hears_is_reported_even_when_unsupported():
    info = _info_for({}, automatic_captions={
        "hi-orig": [{"ext": "json3", "url": "http://x/hi.json3"}],
        "en": [{"ext": "json3", "url": "http://x/en-translated.json3"}],
    })
    assert info.auto_spoken_language == "hi"
    assert info.auto_subtitles == {}  # Hindi is not a language bytx supports; the "en" track is a translation


def test_auto_captions_never_count_as_human_subtitles():
    info = _info_for({}, automatic_captions={"en-orig": [{"ext": "json3", "url": "http://x/en.json3"}]})
    assert info.human_subtitles == {}


def test_parse_json3_captions_joins_pieces_and_skips_scroll_events():
    import json

    data = {"events": [
        {"tStartMs": 0, "dDurationMs": 2500, "wpWinPosId": 1},  # a window definition, no text
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [{"utf8": "hello"}, {"utf8": " world", "tOffsetMs": 400}]},
        {"tStartMs": 4000, "dDurationMs": 500, "segs": [{"utf8": "\n"}]},  # scroll marker
        {"tStartMs": 4500, "dDurationMs": 1500, "segs": [{"utf8": "  second\ncue  "}]},
        {"dDurationMs": 100, "segs": [{"utf8": "no start"}]},
    ]}

    segments = fetch.parse_json3_captions(json.dumps(data))

    assert [(s.idx, s.start_s, s.end_s, s.text) for s in segments] == [
        (0, 1.0, 4.0, "hello world"),
        (1, 4.5, 6.0, "second cue"),
    ]


def test_fetch_auto_caption_segments_downloads_and_parses(monkeypatch):
    class Resp:
        text = '{"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hi"}]}]}'

        def raise_for_status(self):
            pass

    monkeypatch.setattr(fetch.httpx, "get", lambda url, **kw: Resp())
    assert [s.text for s in fetch.fetch_auto_caption_segments("http://x/a.json3")] == ["hi"]

