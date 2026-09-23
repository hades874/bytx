from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bytx.textio import configure_utf8_io  # noqa: E402

configure_utf8_io()
