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


def test_append_entry_records_the_language(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="en1", language="en"))
    ledger.append_entry(path, make_video(id="bn1", language="bn"))
    ledger.append_entry(path, make_video(id="unk1"))

    content = path.read_text(encoding="utf-8")
    entries = {e.split("\n", 1)[0]: e for e in content.strip().split("\n\n")}
    assert "  language: en\n" in entries["- id: en1"] + "\n"
    assert "  language: bn\n" in entries["- id: bn1"] + "\n"
    assert "  language: unknown\n" in entries["- id: unk1"] + "\n"


def test_ledger_with_language_lines_and_old_entries_without_them_load_together(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    path.write_text(
        "- id: old1\n  channel: C\n  title: T\n  url: u\n  date: 2026-09-24\n\n",  # pre-language format
        encoding="utf-8",
    )
    ledger.append_entry(path, make_video(id="new1", language="en"))

    assert ledger.load_seen_ids(path) == {"old1", "new1"}


def test_facebook_ids_roundtrip_next_to_youtube_ones(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    ledger.append_entry(path, make_video(id="abc123"))
    ledger.append_entry(
        path,
        make_video(id="fb_123456789012345", url="https://www.facebook.com/watch/?v=123456789012345",
                   platform="facebook"),
    )

    assert ledger.load_seen_ids(path) == {"abc123", "fb_123456789012345"}
    assert "url: https://www.facebook.com/watch/?v=123456789012345" in path.read_text(encoding="utf-8")


def test_remove_entry_deletes_only_that_video(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    for vid in ("aaa", "bbb", "ccc"):
        ledger.append_entry(path, make_video(id=vid, url=f"https://youtu.be/{vid}"))

    assert ledger.remove_entry(path, "bbb") is True

    assert ledger.load_seen_ids(path) == {"aaa", "ccc"}
    text = path.read_text(encoding="utf-8")
    assert "youtu.be/bbb" not in text and "youtu.be/ccc" in text
    ledger.append_entry(path, make_video(id="bbb", url="https://youtu.be/bbb"))  # can be recorded again
    assert ledger.load_seen_ids(path) == {"aaa", "bbb", "ccc"}


def test_remove_entry_without_an_entry_changes_nothing(tmp_path):
    path = tmp_path / "TRANSCRIBED_LEDGER.md"
    assert ledger.remove_entry(path, "zzz") is False
    assert not path.exists()
    ledger.append_entry(path, make_video(id="aaa"))
    before = path.read_text(encoding="utf-8")
    assert ledger.remove_entry(path, "zzz") is False
    assert path.read_text(encoding="utf-8") == before
