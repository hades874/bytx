"""Writers that turn stored segments into plain text, SRT, VTT, and JSON
files on disk, one set per video under data/videos/<id>/.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from . import store

_ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_FOLDER_NAME_LEN = 150


def _srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def to_txt(segments: Sequence[store.Segment]) -> str:
    """One line per segment. Simple and greppable; `bytx show` can render a
    more reading-friendly paragraph form if desired later.
    """
    return "\n".join(s.text for s in segments) + ("\n" if segments else "")


def to_srt(segments: Sequence[store.Segment]) -> str:
    blocks = []
    for i, s in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_srt_timestamp(s.start_s)} --> {_srt_timestamp(s.end_s)}\n{s.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(segments: Sequence[store.Segment]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_vtt_timestamp(s.start_s)} --> {_vtt_timestamp(s.end_s)}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def to_json(
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
) -> str:
    payload = {
        "id": video.id,
        "url": video.url,
        "title": video.title,
        "channel": video.channel,
        "channel_id": video.channel_id,
        "upload_date": video.upload_date,
        "duration": video.duration,
        "source": video.source,
        "asr_model": video.asr_model,
        "summary": summary,
        "segments": [
            {"start": s.start_s, "end": s.end_s, "text": s.text} for s in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


FORMATS = {"txt", "srt", "vtt", "json"}


def render(
    fmt: str,
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
) -> str:
    if fmt == "txt":
        return to_txt(segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(video, segments, summary=summary)
    raise ValueError(f"Unknown export format {fmt!r}. Known: {sorted(FORMATS)}")


def video_dir(videos_root: Path, video_id: str) -> Path:
    return videos_root / video_id


def write_all(
    videos_root: Path,
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
) -> dict[str, Path]:
    """Writes transcript.txt/.srt/.vtt/.json for a video. Returns a dict of
    format -> written path.
    """
    out_dir = video_dir(videos_root, video.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for fmt, filename in (
        ("txt", "transcript.txt"),
        ("srt", "transcript.srt"),
        ("vtt", "transcript.vtt"),
        ("json", "transcript.json"),
    ):
        path = out_dir / filename
        path.write_text(render(fmt, video, segments, summary=summary), encoding="utf-8")
        written[fmt] = path
    return written


def write_info_json(videos_root: Path, video: store.Video) -> Path:
    out_dir = video_dir(videos_root, video.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "info.json"
    payload = {
        "id": video.id,
        "url": video.url,
        "title": video.title,
        "channel": video.channel,
        "channel_id": video.channel_id,
        "upload_date": video.upload_date,
        "duration": video.duration,
        "description": video.description,
        "source": video.source,
        "asr_model": video.asr_model,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary_md(videos_root: Path, video: store.Video, summary: str) -> Path:
    out_dir = video_dir(videos_root, video.id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.md"
    title = video.title or video.id
    content = f"# {title}\n\n{video.url}\n\n{summary}\n"
    path.write_text(content, encoding="utf-8")
    return path


def _clean_component(value: str) -> str:
    cleaned = _ILLEGAL_CHARS_RE.sub(" ", value)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.rstrip(". ")


def sanitize_folder_name(channel: str | None, title: str | None, video_id: str) -> str:
    """Builds a filesystem-safe "<channel> - <title>" folder name.

    Falls back to video_id for either component if it's missing or becomes
    empty after cleaning. The combined name is truncated (not per-field, so
    a long title doesn't starve the channel name) to leave headroom under
    Windows' 260-char MAX_PATH given yt-transcribed-files/<name>/transcript.txt.
    """
    clean_channel = _clean_component(channel) if channel else ""
    clean_title = _clean_component(title) if title else ""
    clean_channel = clean_channel or video_id
    clean_title = clean_title or video_id

    combined = f"{clean_channel} - {clean_title}"
    if len(combined) > _MAX_FOLDER_NAME_LEN:
        combined = combined[:_MAX_FOLDER_NAME_LEN].rstrip(". ")
    return combined


def _find_existing_folder(root: Path, video_id: str) -> Path | None:
    for child in root.iterdir():
        marker = child / ".video_id"
        if child.is_dir() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == video_id:
            return child
    return None


def write_transcript_copy(
    root: Path, video: store.Video, segments: Sequence[store.Segment]
) -> Path:
    """Writes a human-facing plain-text transcript copy under
    root/"<channel> - <title>"/transcript.txt, alongside a hidden
    .video_id marker used to tell an idempotent re-run of the same video
    (overwrite in place) apart from a genuine folder-name collision between
    two different videos (which gets a " (<video_id>)" suffix instead).
    """
    root.mkdir(parents=True, exist_ok=True)
    # Reuse this video's folder even if its title/channel changed since the
    # last run, so a rename upstream doesn't leave an orphaned copy behind.
    out_dir = _find_existing_folder(root, video.id)
    if out_dir is None:
        name = sanitize_folder_name(video.channel, video.title, video.id)
        out_dir = root / name
        if out_dir.exists():
            out_dir = root / f"{name} ({video.id})"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.txt").write_text(to_txt(segments), encoding="utf-8")
    (out_dir / ".video_id").write_text(video.id, encoding="utf-8")
    return out_dir
