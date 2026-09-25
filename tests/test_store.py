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


# --- language column & schema migration ------------------------------------------

_V1_VIDEOS_TABLE = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (1);
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    channel TEXT,
    channel_id TEXT,
    upload_date TEXT,
    duration REAL,
    description TEXT,
    source TEXT,
    asr_model TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT INTO videos (id, url, title, source, asr_model, status) VALUES
    ('asr1', 'u1', 'Transcribed by ASR', 'asr', 'tugstugi-medium-bn', 'summarized'),
    ('subs1', 'u2', 'Human subtitles', 'human_subs', NULL, 'indexed'),
    ('queued1', 'u3', 'Not processed yet', NULL, NULL, 'queued'),
    ('failed1', 'u4', 'Failed before transcribing', NULL, NULL, 'failed');
"""


def _v1_database(path):
    """A library.db exactly as bytx created it before English support."""
    c = sqlite3.connect(str(path))
    c.executescript(_V1_VIDEOS_TABLE)
    c.commit()
    c.close()


def test_new_database_is_created_at_the_current_schema_version(conn):
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == store.SCHEMA_VERSION == 3
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(videos)")}
    assert {"language", "platform"} <= columns


def test_v1_database_is_migrated_and_keeps_its_rows(tmp_path):
    path = tmp_path / "library.db"
    _v1_database(path)

    c = store.connect(path)

    assert c.execute("SELECT version FROM schema_version").fetchone()[0] == store.SCHEMA_VERSION
    assert {v.id for v in store.list_videos(c)} == {"asr1", "subs1", "queued1", "failed1"}
    assert store.get_video(c, "asr1").title == "Transcribed by ASR"


def test_migration_marks_every_existing_video_bangla(tmp_path):
    """Before English support bytx only handled Bangla, queued and failed rows
    included. NULL now means "detect at run time", so none may be left NULL or
    an old queue would be re-detected."""
    path = tmp_path / "library.db"
    _v1_database(path)

    c = store.connect(path)

    languages = {v.id: v.language for v in store.list_videos(c)}
    assert languages == {"asr1": "bn", "subs1": "bn", "queued1": "bn", "failed1": "bn"}


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "library.db"
    _v1_database(path)
    store.connect(path).close()
    c = store.connect(path)  # second open must not try to add the column again
    store.set_video_language(c, "queued1", "en")
    c.close()

    c = store.connect(path)  # ... and must not redo the backfill over a later choice

    assert store.get_video(c, "queued1").language == "en"
    assert c.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_migrated_database_accepts_new_videos_with_a_language(tmp_path):
    path = tmp_path / "library.db"
    _v1_database(path)
    c = store.connect(path)

    store.upsert_video(c, make_video("fresh", language="en"))

    assert store.get_video(c, "fresh").language == "en"


def test_video_language_defaults_to_none(conn):
    store.upsert_video(conn, make_video())
    assert store.get_video(conn, "abc123").language is None


def test_upsert_keeps_the_stored_language_when_metadata_is_refreshed(conn):
    store.upsert_video(conn, make_video(language="en"))
    store.upsert_video(conn, make_video(title="Refreshed title"))  # language=None, like transcribe_one
    v = store.get_video(conn, "abc123")
    assert (v.title, v.language) == ("Refreshed title", "en")


def test_upsert_fills_in_a_missing_language_but_never_overwrites_one(conn):
    store.upsert_video(conn, make_video())
    store.upsert_video(conn, make_video(language="bn"))
    assert store.get_video(conn, "abc123").language == "bn"
    store.upsert_video(conn, make_video(language="en"))
    assert store.get_video(conn, "abc123").language == "bn"


def test_set_video_language(conn):
    store.upsert_video(conn, make_video())
    store.set_video_language(conn, "abc123", "en")
    assert store.get_video(conn, "abc123").language == "en"


def test_list_videos_filters_by_language(conn):
    store.upsert_video(conn, make_video("e1", language="en"))
    store.upsert_video(conn, make_video("e2", language="en"))
    store.upsert_video(conn, make_video("b1", language="bn"))
    store.upsert_video(conn, make_video("u1"))

    assert {v.id for v in store.list_videos(conn, language="en")} == {"e1", "e2"}
    assert {v.id for v in store.list_videos(conn, language="bn")} == {"b1"}
    assert len(store.list_videos(conn)) == 4


def test_fts_search_is_case_insensitive_for_english(conn):
    store.upsert_video(conn, make_video())
    store.replace_segments(conn, "abc123", [
        store.Segment(idx=0, start_s=0, end_s=5, text="Welcome to the Kubernetes tutorial"),
    ])
    hits = store.search_fts(conn, "kubernetes")
    assert len(hits) == 1 and hits[0].video_id == "abc123"


def test_fts_search_finds_english_word_stems_by_substring(conn):
    store.upsert_video(conn, make_video())
    store.replace_segments(conn, "abc123", [
        store.Segment(idx=0, start_s=0, end_s=5, text="We are transcribing videos today"),
    ])
    assert len(store.search_fts(conn, "transcrib")) == 1


def test_fts_search_covers_both_languages_in_one_library(conn):
    store.upsert_video(conn, make_video("en1", language="en"))
    store.upsert_video(conn, make_video("bn1", language="bn"))
    store.replace_segments(conn, "en1", [store.Segment(idx=0, start_s=0, end_s=5, text="visa interview tips")])
    store.replace_segments(conn, "bn1", [store.Segment(idx=0, start_s=0, end_s=5, text="ভিসা ইন্টারভিউ টিপস")])

    assert [h.video_id for h in store.search_fts(conn, "visa")] == ["en1"]
    assert [h.video_id for h in store.search_fts(conn, "ভিসা")] == ["bn1"]


# --- platform (schema v3) ------------------------------------------------------------

def test_migration_marks_every_existing_video_youtube(tmp_path):
    """Before Facebook support bytx only handled YouTube, so the column default
    is right for every row that predates it."""
    path = tmp_path / "library.db"
    _v1_database(path)

    c = store.connect(path)

    assert {v.id: v.platform for v in store.list_videos(c)} == {
        "asr1": "youtube", "subs1": "youtube", "queued1": "youtube", "failed1": "youtube",
    }


_V2_VIDEOS_TABLE = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version(version) VALUES (2);
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    channel TEXT,
    channel_id TEXT,
    upload_date TEXT,
    duration REAL,
    description TEXT,
    source TEXT,
    asr_model TEXT,
    language TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
INSERT INTO videos (id, url, title, language, status) VALUES
    ('en1', 'u1', 'English one', 'en', 'summarized'),
    ('bn1', 'u2', 'Bangla one', 'bn', 'queued');
"""


def test_v2_database_gains_the_platform_column_and_keeps_its_rows(tmp_path):
    """A library.db exactly as bytx wrote it after English support, before Facebook."""
    path = tmp_path / "library.db"
    c = sqlite3.connect(str(path))
    c.executescript(_V2_VIDEOS_TABLE)
    c.commit()
    c.close()

    c = store.connect(path)

    assert c.execute("SELECT version FROM schema_version").fetchone()[0] == store.SCHEMA_VERSION
    assert {v.id: (v.platform, v.language) for v in store.list_videos(c)} == {
        "en1": ("youtube", "en"), "bn1": ("youtube", "bn"),
    }
    store.upsert_video(c, make_video("fb_123456789012345", platform="facebook"))  # and takes new rows
    assert store.get_video(c, "fb_123456789012345").platform == "facebook"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("first\nsecond", "first"),
        ("\n  \n  padded  \nnext", "padded"),
        ("", None),
        ("   \n ", None),
        (None, None),
    ],
)
def test_first_line(text, expected):
    assert store.first_line(text) == expected


def test_video_platform_defaults_to_youtube_and_is_stored(conn):
    store.upsert_video(conn, make_video())
    store.upsert_video(conn, make_video("fb_123456789012345", platform="facebook"))
    assert store.get_video(conn, "abc123").platform == "youtube"
    assert store.get_video(conn, "fb_123456789012345").platform == "facebook"


def test_upsert_keeps_the_stored_platform_when_metadata_is_refreshed(conn):
    store.upsert_video(conn, make_video("fb_123456789012345", platform="facebook"))
    store.upsert_video(conn, make_video("fb_123456789012345", title="Refreshed"))  # platform defaulted
    v = store.get_video(conn, "fb_123456789012345")
    assert (v.title, v.platform) == ("Refreshed", "facebook")


def test_list_videos_filters_by_platform(conn):
    store.upsert_video(conn, make_video("aaaaaaaaaaa"))
    store.upsert_video(conn, make_video("fb_123456789012345", platform="facebook"))
    assert [v.id for v in store.list_videos(conn, platform="facebook")] == ["fb_123456789012345"]
    assert [v.id for v in store.list_videos(conn, platform="youtube")] == ["aaaaaaaaaaa"]
    assert len(store.list_videos(conn)) == 2
