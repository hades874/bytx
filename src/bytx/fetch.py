"""YouTube interaction via yt-dlp: expanding URLs/playlists/channels/text
files into individual video URLs, fetching metadata, downloading audio, and
fetching uploader-made (human) Bangla subtitles when available.

Only uploader subtitles (yt-dlp's `subtitles`, never `automatic_captions`)
are treated as "human subs" — YouTube's auto-generated captions are lower
quality than our ASR and are intentionally ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import yt_dlp

from . import store
from .config import Config

HUMAN_SUB_LANG_PREFIXES = ("bn",)

_CHANNEL_ROOT_RE = re.compile(
    r"^https?://(www\.)?youtube\.com/(@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)/?$"
)
_KNOWN_TABS = ("videos", "shorts", "streams", "playlists")

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url: str) -> str | None:
    """Pulls the 11-character YouTube video id out of a URL, if present,
    without any network call. Used to dedupe already-queued videos cheaply
    during `bytx add`.
    """
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


@dataclass
class VideoInfo:
    id: str
    url: str
    title: str | None
    channel: str | None
    channel_id: str | None
    upload_date: str | None
    duration: float | None
    description: str | None
    human_subtitle_url: str | None = None
    human_subtitle_lang: str | None = None


def normalize_channel_url(url: str) -> str:
    """A bare channel URL (no tab) pulls in every tab (Videos, Shorts,
    Live...), which usually duplicates content. If we're given a bare
    channel root, point it at the Videos tab explicitly.
    """
    if _CHANNEL_ROOT_RE.match(url):
        return url.rstrip("/") + "/videos"
    return url


def _base_ydl_opts(cfg: Config) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }
    if cfg.fetch.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.fetch.cookies_from_browser,)
    if cfg.fetch.sleep_interval_s:
        opts["sleep_interval"] = cfg.fetch.sleep_interval_s
        opts["max_sleep_interval"] = cfg.fetch.max_sleep_interval_s
    return opts


def _flatten_entries(info: dict) -> list[str]:
    urls: list[str] = []
    entries = info.get("entries")
    if entries is None:
        webpage_url = info.get("webpage_url") or info.get("url")
        return [webpage_url] if webpage_url else []
    for entry in entries:
        if entry is None:
            continue
        if entry.get("entries") is not None:
            urls.extend(_flatten_entries(entry))
            continue
        vid = entry.get("id")
        url = entry.get("url") or entry.get("webpage_url")
        if url and not url.startswith("http"):
            url = None
        if not url and vid:
            url = f"https://www.youtube.com/watch?v={vid}"
        if url:
            urls.append(url)
    return urls


def expand_to_video_urls(target: str, cfg: Config) -> list[str]:
    """Expands a single video URL, playlist URL, channel URL, or a local
    text file (one target per line, '#' comments and blank lines skipped)
    into a flat list of individual video URLs. Uses flat extraction so it
    does not fetch full metadata per video at this stage.
    """
    path = Path(target)
    if path.exists() and path.is_file():
        urls: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.extend(expand_to_video_urls(line, cfg))
        return urls

    normalized = normalize_channel_url(target)
    opts = _base_ydl_opts(cfg)
    opts["extract_flat"] = "in_playlist"
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(normalized, download=False)
    if info is None:
        return []
    return [u for u in _flatten_entries(info) if u]


def _pick_human_bn_subtitle(info: dict) -> tuple[str | None, str | None]:
    subs = info.get("subtitles") or {}
    for lang, formats in subs.items():
        if not lang.lower().startswith(HUMAN_SUB_LANG_PREFIXES):
            continue
        vtt = next((f for f in formats if f.get("ext") == "vtt"), None)
        chosen = vtt or (formats[0] if formats else None)
        if chosen:
            return chosen["url"], lang
    return None, None


def get_video_info(url: str, cfg: Config) -> VideoInfo:
    opts = _base_ydl_opts(cfg)
    opts["extract_flat"] = False
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise RuntimeError(f"yt-dlp returned no info for {url}")

    sub_url, sub_lang = _pick_human_bn_subtitle(info)
    return VideoInfo(
        id=info["id"],
        url=info.get("webpage_url", url),
        title=info.get("title"),
        channel=info.get("channel") or info.get("uploader"),
        channel_id=info.get("channel_id"),
        upload_date=info.get("upload_date"),
        duration=info.get("duration"),
        description=info.get("description"),
        human_subtitle_url=sub_url,
        human_subtitle_lang=sub_lang,
    )


def download_audio(url: str, out_dir: Path, cfg: Config) -> Path:
    """Downloads the best available audio-only stream, no transcoding.
    faster-whisper decodes most containers directly via PyAV/ffmpeg, so we
    skip postprocessing to save time and disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _base_ydl_opts(cfg)
    opts.update(
        {
            "skip_download": False,
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
            "postprocessors": [],
        }
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"yt-dlp failed to download audio for {url}")
        filename = ydl.prepare_filename(info)
    return Path(filename)


_VTT_TIME_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[.,](\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _vtt_ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_vtt(text: str) -> list[store.Segment]:
    """Minimal WebVTT parser: extracts (start, end, text) cues. Ignores
    headers, NOTE blocks, and cue settings after the timestamp line, and
    strips inline tags like <b> or karaoke timestamps.
    """
    segments: list[store.Segment] = []
    lines = text.splitlines()
    i = 0
    idx = 0
    while i < len(lines):
        m = _VTT_TIME_RE.search(lines[i])
        if m:
            start = _vtt_ts_to_seconds(*m.groups()[0:4])
            end = _vtt_ts_to_seconds(*m.groups()[4:8])
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != "" and not _VTT_TIME_RE.search(lines[i]):
                text_lines.append(_TAG_RE.sub("", lines[i]).strip())
                i += 1
            cue_text = " ".join(t for t in text_lines if t)
            if cue_text:
                segments.append(store.Segment(idx=idx, start_s=start, end_s=end, text=cue_text))
                idx += 1
        else:
            i += 1
    return segments


def fetch_human_subtitle_segments(sub_url: str) -> list[store.Segment]:
    resp = httpx.get(sub_url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return parse_vtt(resp.text)
