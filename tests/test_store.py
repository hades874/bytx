from __future__ import annotations

import sqlite3

import pytest

from bytx import store


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    store.init_db(c)
    yield c
    c.close()


def make_video(video_id: str = "abc123", **overrides) -> store.Video:
    defaults = dict(id=video_id, url=f"https://youtu.be/{video_id}", title="Test video")
    defaults.update(overrides)
    return store.Video(**defaults)


def test_upsert_and_get_video(conn):
    store.upsert_video(conn, make_video())
    v = store.get_video(conn, "abc123")
    assert v is not None
    assert v.title == "Test video"
    assert v.status == "queued"


def test_upsert_video_partial_update_preserves_existing_fields(conn):
    store.upsert_video(conn, make_video(channel="Original Channel"))
    # Re-upsert with title=None should not clobber the existing title/channel
    store.upsert_video(conn, store.Video(id="abc123", url="https://youtu.be/abc123"))
    v = store.get_video(conn, "abc123")
    assert v.channel == "Original Channel"


def test_video_exists(conn):
    assert not store.video_exists(conn, "abc123")
    store.upsert_video(conn, make_video())
    assert store.video_exists(conn, "abc123")


def test_set_video_status_and_attempts(conn):
    store.upsert_video(conn, make_video())
    store.set_video_status(conn, "abc123", "failed", error="boom", increment_attempts=True)
    v = store.get_video(conn, "abc123")
    assert v.status == "failed"
    assert v.error == "boom"
    assert v.attempts == 1

    store.set_video_status(conn, "abc123", "failed", error="boom again", increment_attempts=True)
    v = store.get_video(conn, "abc123")
    assert v.attempts == 2


def test_list_videos_filters(conn):
    store.upsert_video(conn, make_video("v1", channel_id="c1", status="queued"))
    store.upsert_video(conn, make_video("v2", channel_id="c2", status="transcribed"))
    store.set_video_status(conn, "v2", "transcribed")

    all_videos = store.list_videos(conn)
    assert {v.id for v in all_videos} == {"v1", "v2"}

    queued = store.list_videos(conn, status="queued")
    assert [v.id for v in queued] == ["v1"]

    by_channel = store.list_videos(conn, channel_id="c2")
    assert [v.id for v in by_channel] == ["v2"]


def test_replace_and_get_segments(conn):
    store.upsert_video(conn, make_video())
    segs = [
        store.Segment(idx=0, start_s=0.0, end_s=2.5, text="hello"),
        store.Segment(idx=1, start_s=2.5, end_s=5.0, text="world"),
    ]
    store.replace_segments(conn, "abc123", segs)
    got = store.get_segments(conn, "abc123")
    assert [s.text for s in got] == ["hello", "world"]
    assert got[0].start_s == 0.0

    # Replacing again should fully overwrite, not append
    store.replace_segments(conn, "abc123", [store.Segment(idx=0, start_s=0.0, end_s=1.0, text="only one")])
    got = store.get_segments(conn, "abc123")
    assert len(got) == 1
    assert got[0].text == "only one"


def test_fts_trigram_matches_bangla_substring(conn):
    """Trigram tokenizer should find a root word even when the stored text
    has it wrapped in Bangla suffixes/other words (no word-boundary requirement).
    """
    store.upsert_video(conn, make_video(title="Bangla lecture"))
    segs = [
        store.Segment(idx=0, start_s=0.0, end_s=3.0, text="বাংলাদেশের ইতিহাস অনেক পুরনো"),
        store.Segment(idx=1, start_s=3.0, end_s=6.0, text="আজকে আমরা গণিত নিয়ে কথা বলব"),
    ]
    store.replace_segments(conn, "abc123", segs)

    hits = store.search_fts(conn, "বাংলাদেশ")
    assert len(hits) == 1
    assert hits[0].video_id == "abc123"
    assert hits[0].start_s == 0.0

    hits2 = store.search_fts(conn, "গণিত")
    assert len(hits2) == 1
    assert hits2[0].start_s == 3.0

    hits3 = store.search_fts(conn, "নেইএমনকিছুনেই")
    assert hits3 == []


def test_fts_search_rejects_short_query(conn):
    with pytest.raises(ValueError):
        store.search_fts(conn, "ab")


def test_fts_search_filters_by_channel(conn):
    store.upsert_video(conn, make_video("v1", channel="Channel A"))
    store.upsert_video(conn, make_video("v2", channel="Channel B"))
    store.replace_segments(conn, "v1", [store.Segment(idx=0, start_s=0, end_s=1, text="unique term alpha")])
    store.replace_segments(conn, "v2", [store.Segment(idx=0, start_s=0, end_s=1, text="unique term alpha")])

    hits = store.search_fts(conn, "unique term", channel="Channel A")
    assert [h.video_id for h in hits] == ["v1"]


def test_fts_sync_on_delete_and_update(conn):
    store.upsert_video(conn, make_video())
    store.replace_segments(conn, "abc123", [store.Segment(idx=0, start_s=0, end_s=1, text="findme term")])
    assert len(store.search_fts(conn, "findme")) == 1

    # replace_segments deletes old rows and re-inserts; FTS should follow via triggers
    store.replace_segments(conn, "abc123", [store.Segment(idx=0, start_s=0, end_s=1, text="gone now")])
    assert store.search_fts(conn, "findme") == []
    assert len(store.search_fts(conn, "gone now")) == 1


def test_floats_blob_roundtrip():
    values = [0.1, -1.5, 3.333333, 0.0]
    blob = store.floats_to_blob(values)
    assert isinstance(blob, bytes)
    restored = store.blob_to_floats(blob)
    assert len(restored) == len(values)
    for a, b in zip(values, restored, strict=True):
        assert abs(a - b) < 1e-6


def test_replace_and_get_chunks(conn):
    store.upsert_video(conn, make_video())
    emb = store.floats_to_blob([1.0, 2.0, 3.0])
    chunks = [
        store.Chunk(start_s=0, end_s=75, text="chunk one", embedding=emb, embed_model="bge-m3"),
        store.Chunk(start_s=60, end_s=135, text="chunk two", embedding=emb, embed_model="bge-m3"),
    ]
    store.replace_chunks(conn, "abc123", chunks)
    got = store.get_chunks(conn, "abc123")
    assert len(got) == 2
    assert got[0].embed_model == "bge-m3"
    assert store.blob_to_floats(got[0].embedding) == [1.0, 2.0, 3.0]


def test_get_chunks_all_videos(conn):
    store.upsert_video(conn, make_video("v1"))
    store.upsert_video(conn, make_video("v2"))
    store.replace_chunks(conn, "v1", [store.Chunk(start_s=0, end_s=1, text="a")])
    store.replace_chunks(conn, "v2", [store.Chunk(start_s=0, end_s=1, text="b")])
    assert len(store.get_chunks(conn)) == 2


def test_summary_upsert_and_get(conn):
    store.upsert_video(conn, make_video())
    assert store.get_summary(conn, "abc123") is None
    store.upsert_summary(conn, "abc123", "gemma3:4b", "প্রথম সারাংশ")
    assert store.get_summary(conn, "abc123") == "প্রথম সারাংশ"

    store.upsert_summary(conn, "abc123", "gemma3:4b", "দ্বিতীয় সারাংশ")
    assert store.get_summary(conn, "abc123") == "দ্বিতীয় সারাংশ"


def test_video_cascade_delete_removes_segments_and_chunks(conn):
    store.upsert_video(conn, make_video())
    store.replace_segments(conn, "abc123", [store.Segment(idx=0, start_s=0, end_s=1, text="hi")])
    store.replace_chunks(conn, "abc123", [store.Chunk(start_s=0, end_s=1, text="hi")])
    conn.execute("DELETE FROM videos WHERE id = ?", ("abc123",))
    conn.commit()
    assert store.get_segments(conn, "abc123") == []
    assert store.get_chunks(conn, "abc123") == []
