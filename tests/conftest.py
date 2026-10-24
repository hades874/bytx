from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bytx.textio import configure_utf8_io  # noqa: E402

configure_utf8_io()


@pytest.fixture(autouse=True)
def _isolated_data_home(tmp_path, monkeypatch):
    """No test may read or write the real ~/bytx."""
    monkeypatch.setenv("BYTX_HOME", str(tmp_path / "bytx-home"))
