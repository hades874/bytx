"""Durable, git-tracked record of transcribed video ids.

The SQLite DB (data/library.db) is gitignored and is the primary source of
truth while it exists, but a wiped data/ dir or a fresh clone would leave it
with no memory of what's already been transcribed. TRANSCRIBED_LEDGER.md is
a plain-text backstop that survives both: append-only, one process writes it
at a time (bytx has no daemon/concurrent-writer mode), so no locking is
needed.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from . import store
from .config import Config

DEFAULT_LEDGER_FILENAME = "TRANSCRIBED_LEDGER.md"

_ID_LINE_RE = re.compile(r"^- id: (\S+)$", re.MULTILINE)
_ENTRY_RE = re.compile(r"^- id: (\S+)\n(?:  .*\n)*\n?", re.MULTILINE)


def ledger_path(cfg: Config) -> Path:
    return cfg.root / DEFAULT_LEDGER_FILENAME


def load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(_ID_LINE_RE.findall(path.read_text(encoding="utf-8")))


def _one_line(value: str) -> str:
    """Collapses whitespace/newlines so a value can't spill onto a new line
    and forge a "- id:" entry in this line-based format."""
    return " ".join(value.split())


def append_entry(path: Path, video: store.Video) -> None:
    """Records a transcribed video. A no-op if its id is already recorded,
    so retries never produce duplicate entries."""
    if video.id in load_seen_ids(path):
        return
    channel = _one_line(video.channel or video.id)
    title = _one_line(video.title or video.id)
    entry = (
        f"- id: {video.id}\n"
        f"  channel: {channel}\n"
        f"  title: {title}\n"
        f"  url: {video.url}\n"
        f"  language: {video.language or 'unknown'}\n"
        f"  date: {date.today().isoformat()}\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)


def remove_entry(path: Path, video_id: str) -> bool:
    """Deletes `video_id`'s entry, so `bytx add` accepts the video again.
    Returns whether there was one."""
    if video_id not in load_seen_ids(path):
        return False
    text = path.read_text(encoding="utf-8")
    path.write_text(
        _ENTRY_RE.sub(lambda m: "" if m.group(1) == video_id else m.group(0), text), encoding="utf-8"
    )
    return True
