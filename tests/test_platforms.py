from __future__ import annotations

import pytest

from bytx import platforms
from bytx.platforms import FACEBOOK, YOUTUBE, VideoRef


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=aaaaaaaaaaa", VideoRef(YOUTUBE, "aaaaaaaaaaa")),
        ("https://www.youtube.com/watch?list=PL1&v=bbbbbbbbbbb", VideoRef(YOUTUBE, "bbbbbbbbbbb")),
        ("https://youtu.be/ccccccccccc?t=5", VideoRef(YOUTUBE, "ccccccccccc")),
        ("https://www.youtube.com/shorts/ddddddddddd", VideoRef(YOUTUBE, "ddddddddddd")),
        ("https://www.facebook.com/watch/?v=123456789012345", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/watch?v=123456789012345&t=9", VideoRef(FACEBOOK, "123456789012345")),
        ("https://m.facebook.com/watch/?v=123456789012345", VideoRef(FACEBOOK, "123456789012345")),
        ("https://web.facebook.com/watch/?v=123456789012345", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/video.php?v=123456789012345", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/SomePage/videos/123456789012345", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/SomePage/videos/123456789012345/", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/SomePage/videos/some-slug/123456789012345/", VideoRef(FACEBOOK, "123456789012345")),
        ("https://www.facebook.com/reel/123456789012345", VideoRef(FACEBOOK, "123456789012345")),
    ],
)
def test_parse_video_ref(url, expected):
    assert platforms.parse_video_ref(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://fb.watch/abcDEF12/",  # opaque short link
        "https://www.facebook.com/share/v/1AbCdEfGh/",  # opaque share link
        "https://www.facebook.com/share/r/1AbCdEfGh/",
        "https://www.facebook.com/SomePage",  # a page, not a video
        "https://www.facebook.com/watch/?v=notanumber",
        "https://www.youtube.com/@SomeChannel",
        "https://vimeo.com/123456789",
        "not a url",
    ],
)
def test_parse_video_ref_is_none_when_the_url_carries_no_id(url):
    assert platforms.parse_video_ref(url) is None


def test_storage_id_leaves_youtube_ids_bare_and_prefixes_facebook_ones():
    assert VideoRef(YOUTUBE, "aaaaaaaaaaa").id == "aaaaaaaaaaa"
    assert VideoRef(FACEBOOK, "123").id == "fb_123"


@pytest.mark.parametrize(
    "ref",
    [
        VideoRef(YOUTUBE, "aaaaaaaaaaa"),
        VideoRef(YOUTUBE, "fb_abcdefgh"),  # a valid YouTube id that happens to start with fb_
        VideoRef(YOUTUBE, "fb_12345678"),  # ... even with only digits after it
        VideoRef(FACEBOOK, "123456789012345"),
    ],
)
def test_ref_from_id_inverts_the_storage_id(ref):
    assert platforms.ref_from_id(ref.id) == ref


def test_platform_of_reads_the_prefix():
    assert platforms.platform_of("fb_123456789012345") == FACEBOOK
    assert platforms.platform_of("aaaaaaaaaaa") == YOUTUBE


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://fb.watch/abc/", FACEBOOK),
        ("https://www.facebook.com/share/v/abc/", FACEBOOK),
        ("https://m.youtube.com/watch?v=aaaaaaaaaaa", YOUTUBE),
        ("https://youtu.be/aaaaaaaaaaa", YOUTUBE),
        ("https://example.com/facebook.com/watch", None),
        ("https://notfacebook.com/x", None),
    ],
)
def test_platform_of_url(url, expected):
    assert platforms.platform_of_url(url) == expected


def test_timestamp_url_per_platform():
    assert platforms.timestamp_url("aaaaaaaaaaa", 75.9) == "https://youtu.be/aaaaaaaaaaa?t=75"
    assert (
        platforms.timestamp_url("fb_123456789012345", 75.9)
        == "https://www.facebook.com/watch/?v=123456789012345&t=75"
    )
