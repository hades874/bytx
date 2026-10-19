"""Remove a video from bytx so it can be transcribed again (see docs/SOP-video-transcription.md).

`bytx add` refuses a video that is in the DB or in TRANSCRIBED_LEDGER.md, and
there is no built-in "remove", so a redo (usually: the video was queued with the
wrong --lang) means clearing every place the video was recorded:

  * the DB row (segments, chunks and the summary go with it via ON DELETE CASCADE)
  * data/videos/<id>/
  * its yt-transcribed-files/<date>/<folder>/ copy, then INDEX.md is rebuilt
  * its entry in TRANSCRIBED_LEDGER.md

Dry run by default; pass --yes to delete. Afterwards re-add with the right language:

    uv run python scripts/redo_video.py <video_id> --yes
    uv run bytx add <url-or-file> --lang en --run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bytx import export, ledger, store  # noqa: E402
from bytx.config import load_config  # noqa: E402
from bytx.textio import configure_utf8_io  # noqa: E402

_ENTRY_RE = re.compile(r"^- id: (\S+)\n(?:  .*\n)*\n?", re.MULTILINE)


def _strip_ledger_entry(text: str, video_id: str) -> str:
    return _ENTRY_RE.sub(lambda m: "" if m.group(1) == video_id else m.group(0), text)


def main() -> int:
    configure_utf8_io()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video_id")
    parser.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = store.connect(cfg.paths.db_path)
    vid = args.video_id

    video = store.get_video(conn, vid)
    export_dir = export.find_transcript_folder(cfg.paths.transcribed_files_dir, vid)
    videos_dir = export.video_dir(cfg.paths.videos_dir, vid)
    ledger_file = ledger.ledger_path(cfg)
    in_ledger = vid in ledger.load_seen_ids(ledger_file)

    print(f"video {vid}: {video.title if video else '(not in DB)'}  [{video.language if video else '-'}]")
    print(f"  DB row:        {'yes' if video else 'no'}")
    print(f"  data dir:      {videos_dir if videos_dir.is_dir() else 'none'}")
    print(f"  shared folder: {export_dir or 'none'}")
    print(f"  ledger entry:  {'yes' if in_ledger else 'no'}")
    if not (video or export_dir or videos_dir.is_dir() or in_ledger):
        print("Nothing to remove.")
        return 1
    if not args.yes:
        print("Dry run. Re-run with --yes to delete the above.")
        return 0

    if video:
        conn.execute("DELETE FROM videos WHERE id = ?", (vid,))
        conn.commit()
    if videos_dir.is_dir():
        shutil.rmtree(videos_dir)
    if export_dir:
        shutil.rmtree(export_dir)
        if not any(export_dir.parent.iterdir()):
            export_dir.parent.rmdir()  # the now-empty <date>/ folder
        export.write_index(cfg.paths.transcribed_files_dir)
    if in_ledger:
        text = ledger_file.read_text(encoding="utf-8")
        ledger_file.write_text(_strip_ledger_entry(text, vid), encoding="utf-8")
    print(f"Removed {vid}. Re-add it with `bytx add ... --lang <bn|en>`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
