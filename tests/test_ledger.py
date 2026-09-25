from __future__ import annotations

from bytx import ledger, store


def make_video(**overrides) -> store.Video:
    defaults = dict(
        id="abc123", url="https://youtu.be/abc123", title="বাংলা লেকচার", channel="Test Channel",
    )
    defaults.update(overrides)
    return store.Video(**defaults)


def test_load_seen_ids_returns_empty_set_when_file_missing(tmp_path):
    assert ledger.load_seen_ids(tmp_path / "TRANSCRIBED_LEDGER.md") == set()


def test_append_entry_then_load_roundtrips_id(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="abc123"))
    ledger.append_entry(path, make_video(id="xyz789"))

    assert ledger.load_seen_ids(path) == {"abc123", "xyz789"}
    content = path.read_text(encoding="utf-8")
    assert "channel: Test Channel" in content
    assert "title: বাংলা লেকচার" in content
    assert "url: https://youtu.be/abc123" in content


def test_append_entry_falls_back_to_id_when_channel_or_title_missing(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="abc123", channel=None, title=None))

    content = path.read_text(encoding="utf-8")
    assert "channel: abc123" in content
    assert "title: abc123" in content


def test_load_seen_ids_ignores_non_id_lines(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    path.write_text(
        "- id: abc123\n"
        "  channel: id: not-a-real-id\n"
        "  title: id: also-not-real\n\n",
        encoding="utf-8",
    )
    assert ledger.load_seen_ids(path) == {"abc123"}


def test_append_entry_is_idempotent(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="abc123"))
    ledger.append_entry(path, make_video(id="abc123"))
    assert path.read_text(encoding="utf-8").count("- id: abc123") == 1


def test_append_entry_keeps_multiline_title_on_one_line(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="abc123", title="x\n- id: forged"))
    assert ledger.load_seen_ids(path) == {"abc123"}
