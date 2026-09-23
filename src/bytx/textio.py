"""Windows terminals often default to a legacy codepage (e.g. cp1252) that
cannot print Bangla text, raising UnicodeEncodeError. Call
`configure_utf8_io()` once at process start (CLI entrypoint, test conftest)
to force UTF-8 stdout/stderr regardless of the console's codepage.
"""

from __future__ import annotations

import sys


def configure_utf8_io() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
