"""Which video platform a URL or stored video id belongs to.

YouTube ids are stored bare (11 characters), as they always have been, so
existing rows, ledger entries and export folders stay valid. Facebook ids are
stored as `fb_<numeric id>`: the prefix keeps the id namespaces apart, and unlike
`fb:` it is a legal Windows file name (ids name folders under data/videos/).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

YOUTUBE = "youtube"
FACEBOOK = "facebook"

_FACEBOOK_PREFIX = "fb_"

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

_FACEBOOK_WATCH_PATHS = {"/watch", "/watch/", "/watch/live", "/watch/live/", "/video.php"}
_FACEBOOK_REEL_RE = re.compile(r"^/reel/(\d+)")
_FACEBOOK_VIDEOS_RE = re.compile(r"/videos/(?:[^/]+/)?(\d+)/?$")


@dataclass(frozen=True)
class VideoRef:
    platform: str
    native_id: str

    @property
    def id(self) -> str:
        """The id this video is stored under."""
        return storage_id(self.platform, self.native_id)


def storage_id(platform: str, native_id: str) -> str:
    return _FACEBOOK_PREFIX + native_id if platform == FACEBOOK else native_id


def ref_from_id(video_id: str) -> VideoRef:
    """Inverse of `VideoRef.id`.

    YouTube ids are exactly 11 characters from [A-Za-z0-9_-], so one can start
    with "fb_". Requiring digits after the prefix and a length other than 11
    keeps such an id from being read as Facebook's (real Facebook ids are far
    longer than 8 digits).
    """
    native = video_id[len(_FACEBOOK_PREFIX):]
    if video_id.startswith(_FACEBOOK_PREFIX) and native.isdigit() and len(video_id) != 11:
        return VideoRef(FACEBOOK, native)
    return VideoRef(YOUTUBE, video_id)


def platform_of(video_id: str) -> str:
    return ref_from_id(video_id).platform


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_facebook_url(url: str) -> bool:
    host = _host(url)
    return host == "facebook.com" or host.endswith(".facebook.com") or host == "fb.watch"


def is_youtube_url(url: str) -> bool:
    host = _host(url)
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def platform_of_url(url: str) -> str | None:
    if is_facebook_url(url):
        return FACEBOOK
    if is_youtube_url(url):
        return YOUTUBE
    return None


def parse_video_ref(url: str) -> VideoRef | None:
    """Pulls the platform and native video id out of a URL without any network
    call, or None when the URL doesn't carry one (a bare channel or page, an
    unsupported site, or an opaque short link like fb.watch/xyz or
    facebook.com/share/v/xyz that only resolves by fetching it).
    """
    if is_facebook_url(url):
        native = _facebook_native_id(url)
        return VideoRef(FACEBOOK, native) if native else None
    m = _YOUTUBE_ID_RE.search(url)
    return VideoRef(YOUTUBE, m.group(1)) if m else None


def _facebook_native_id(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.path in _FACEBOOK_WATCH_PATHS:
        vid = (parse_qs(parsed.query).get("v") or [""])[0]
        return vid if vid.isdigit() else None
    m = _FACEBOOK_REEL_RE.match(parsed.path) or _FACEBOOK_VIDEOS_RE.search(parsed.path)
    return m.group(1) if m else None


def timestamp_url(video_id: str, start_s: float) -> str:
    """A link to `video_id` at `start_s`, derived from the stored id alone."""
    ref = ref_from_id(video_id)
    seconds = int(start_s)
    if ref.platform == FACEBOOK:
        return f"https://www.facebook.com/watch/?v={ref.native_id}&t={seconds}"
    return f"https://youtu.be/{ref.native_id}?t={seconds}"
