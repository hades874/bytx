"""Writers that turn stored segments into plain text, SRT, VTT, and JSON
files on disk: a full set per video under data/videos/<id>/, and a
human-facing copy (transcript, timestamps, metadata, quality flags) under
yt-transcribed-files/<date>/<channel> - <title>/ with a generated INDEX.md over all of them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from urllib.parse import quote

from . import store
from .quality import Quality, format_time

_ILLEGAL_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_FOLDER_NAME_LEN = 150
_DATE_DIR_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
INDEX_FILENAME = "INDEX.md"
# A hidden file in each transcript folder that says which video it holds, so a re-run
# finds its folder even after the title changed.
_MARKER = ".video_id"


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


def _segment_json(s: store.Segment) -> dict:
    data: dict = {"start": s.start_s, "end": s.end_s, "text": s.text}
    if s.avg_logprob is not None:
        data["avg_logprob"] = round(s.avg_logprob, 4)
    if s.no_speech_prob is not None:
        data["no_speech_prob"] = round(s.no_speech_prob, 4)
    return data


def to_json(
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
    quality: Quality | None = None,
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
        "language": video.language,
        "summary": summary,
        "segments": [_segment_json(s) for s in segments],
    }
    if quality is not None:
        payload["quality"] = quality.to_dict()
    return json.dumps(payload, ensure_ascii=False, indent=2)


FORMATS = {"txt", "srt", "vtt", "json"}


def render(
    fmt: str,
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
    quality: Quality | None = None,
) -> str:
    if fmt == "txt":
        return to_txt(segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(video, segments, summary=summary, quality=quality)
    raise ValueError(f"Unknown export format {fmt!r}. Known: {sorted(FORMATS)}")


def video_dir(videos_root: Path, video_id: str) -> Path:
    return videos_root / video_id


def write_all(
    videos_root: Path,
    video: store.Video,
    segments: Sequence[store.Segment],
    summary: str | None = None,
    quality: Quality | None = None,
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
        path.write_text(
            render(fmt, video, segments, summary=summary, quality=quality), encoding="utf-8"
        )
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
        "platform": video.platform,
        "source": video.source,
        "asr_model": video.asr_model,
        "language": video.language,
        "transcribed_at": video.transcribed_at,
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


def metadata_dict(
    video: store.Video, quality: Quality | None = None, glossary_replacements: int = 0
) -> dict:
    """What a reader needs to judge a transcript without opening the database:
    where it came from, how it was made, and how far to trust it."""
    return {
        "id": video.id,
        "url": video.url,
        "platform": video.platform,
        "title": video.title or store.first_line(video.description),
        "channel": video.channel,
        "channel_id": video.channel_id,
        "upload_date": video.upload_date,
        "duration": video.duration,
        "language": video.language,
        "source": video.source,
        "asr_model": video.asr_model,
        "transcribed_at": video.transcribed_at,
        "glossary_replacements": glossary_replacements,
        "quality": quality.to_dict() if quality is not None else None,
    }


def _clean_component(value: str) -> str:
    cleaned = _ILLEGAL_CHARS_RE.sub(" ", value)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.rstrip(". ")


def sanitize_folder_name(channel: str | None, title: str | None, video_id: str) -> str:
    """Builds a filesystem-safe "<channel> - <title>" folder name.

    Falls back to video_id for either component if it's missing or becomes
    empty after cleaning. The combined name is truncated (not per-field, so
    a long title doesn't starve the channel name) to leave headroom under
    Windows' 260-char MAX_PATH given
    yt-transcribed-files/<date>/<name>/transcript.txt.
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
    """Finds the folder already holding this video, whichever date folder it
    was filed under (or directly under root, for the pre-date-folder layout).
    A folder is recognised by its .video_id marker, or, for one created by the
    short-lived id-named layout, by being named after the video id."""
    date_dirs = [c for c in root.iterdir() if c.is_dir() and _DATE_DIR_RE.fullmatch(c.name)]
    candidates = [c for c in root.iterdir() if c.is_dir() and c not in date_dirs]
    for date_dir in date_dirs:
        candidates.extend(child for child in date_dir.iterdir() if child.is_dir())
    for child in candidates:
        marker = child / _MARKER
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == video_id:
            return child
    for date_dir in date_dirs:
        if (date_dir / video_id).is_dir():
            return date_dir / video_id
    return None


def find_transcript_folder(root: Path, video_id: str) -> Path | None:
    """The shared-copy folder of `video_id` under `root`, if there is one."""
    return _find_existing_folder(root, video_id) if root.is_dir() else None


def _folder_name(video: store.Video) -> str:
    title = video.title or store.first_line(video.description)
    return sanitize_folder_name(video.channel, title, video.id)


def _unused(path: Path, video_id: str) -> Path:
    """`path`, or with a " (<video_id>)" suffix when it already exists, so two
    different videos with the same channel and title never share a folder."""
    return path.with_name(f"{path.name} ({video_id})") if path.exists() else path


def write_transcript_copy(
    root: Path,
    video: store.Video,
    segments: Sequence[store.Segment],
    quality: Quality | None = None,
    cleaned: Sequence[store.Segment] | None = None,
    glossary_replacements: int = 0,
    today: date | None = None,
) -> Path:
    """Writes the human-facing copy of a transcript to
    root/<YYYY-MM-DD>/"<channel> - <title>"/, where the date is the day it was
    first transcribed (a re-run keeps the original date folder). Files:

    - transcript.txt / .vtt / .json: the raw ASR text, with timestamps and
      per-clip confidence in the .json;
    - transcript.clean.txt: the same text with glossary corrections applied,
      written only when the glossary changed something;
    - metadata.json: title, channel, date, duration, language, source, model
      and the quality assessment.

    A hidden .video_id marker tells an idempotent re-run of the same video
    (overwrite in place, even if its title changed) apart from a genuine
    folder-name collision between two different videos (which gets a
    " (<video_id>)" suffix). A folder from the short-lived id-named layout is
    renamed to the title form.
    """
    root.mkdir(parents=True, exist_ok=True)
    today_dir = root / (today or date.today()).isoformat()
    out_dir = _find_existing_folder(root, video.id)
    if out_dir is None:
        out_dir = _unused(today_dir / _folder_name(video), video.id)
    elif out_dir.name == video.id and not (out_dir / _MARKER).is_file():
        target = _unused(out_dir.parent / _folder_name(video), video.id)
        if target != out_dir:
            out_dir.rename(target)
        out_dir = target
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / _MARKER).write_text(video.id, encoding="utf-8")
    (out_dir / "transcript.txt").write_text(to_txt(segments), encoding="utf-8")
    (out_dir / "transcript.vtt").write_text(to_vtt(segments), encoding="utf-8")
    (out_dir / "transcript.json").write_text(
        to_json(video, segments, quality=quality), encoding="utf-8"
    )
    clean_path = out_dir / "transcript.clean.txt"
    if cleaned is not None and glossary_replacements:
        clean_path.write_text(to_txt(cleaned), encoding="utf-8")
    else:
        clean_path.unlink(missing_ok=True)  # a stale one would contradict the raw text
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata_dict(video, quality, glossary_replacements), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def _md_cell(value: object) -> str:
    text = " ".join(str(value if value is not None else "").split())
    return text.replace("|", "\\|")


def write_index(root: Path) -> Path:
    """Regenerates root/INDEX.md: one row per transcript folder, read from its
    metadata.json, newest date first, with the quality flags that say which
    ones to double-check before citing."""
    rows: list[tuple[str, str, dict]] = []
    if root.is_dir():
        for date_dir in root.iterdir():
            if not (date_dir.is_dir() and _DATE_DIR_RE.fullmatch(date_dir.name)):
                continue
            for meta_path in date_dir.glob("*/metadata.json"):
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                rows.append((date_dir.name, meta_path.parent.name, meta))
    rows.sort(key=lambda r: (r[0], r[2].get("channel") or "", r[1]), reverse=True)

    lines = [
        "# Transcripts",
        "",
        "Generated by bytx; do not edit. Flags mark transcripts to double-check before citing.",
        "",
        "| Transcribed | Video | Channel | Title | Lang | Length | Words/min | Flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for day, folder, meta in rows:
        quality = meta.get("quality") or {}
        wpm = quality.get("words_per_minute")
        duration = meta.get("duration")
        cells = (
            day,
            f"[{meta.get('id')}]({quote(day)}/{quote(folder)}/transcript.txt)",
            meta.get("channel"),
            meta.get("title"),
            meta.get("language"),
            format_time(duration) if duration else "",
            f"{wpm:.0f}" if wpm else "",
            "; ".join(quality.get("flags") or []),
        )
        lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
    path = root / INDEX_FILENAME
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
