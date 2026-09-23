"""SQLite storage layer: schema, migrations, CRUD, and FTS5 (trigram) search.

Bangla words carry suffixes (e.g. বাংলাদেশ -> বাংলাদেশের), so keyword search
uses FTS5's `trigram` tokenizer, which matches any substring rather than
whole tokens. This means a search for the root word still finds inflected
forms, at the cost of some false positives on short queries (mitigated by
requiring queries of at least 3 characters).
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
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

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    content='segments',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    embed_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_video_id ON chunks(video_id);

CREATE TABLE IF NOT EXISTS summaries (
    video_id TEXT PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
    model TEXT,
    summary TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


@dataclass
class Video:
    id: str
    url: str
    title: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    upload_date: str | None = None
    duration: float | None = None
    description: str | None = None
    source: str | None = None
    asr_model: str | None = None
    status: str = "queued"
    error: str | None = None
    attempts: int = 0
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Video:
        return cls(**{k: row[k] for k in row.keys()})


@dataclass
class Segment:
    idx: int
    start_s: float
    end_s: float
    text: str
    video_id: str | None = None
    id: int | None = None


@dataclass
class Chunk:
    start_s: float
    end_s: float
    text: str
    video_id: str | None = None
    embedding: bytes | None = None
    embed_model: str | None = None
    id: int | None = None


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# --- Videos ---------------------------------------------------------------

def upsert_video(conn: sqlite3.Connection, video: Video) -> None:
    conn.execute(
        """
        INSERT INTO videos (id, url, title, channel, channel_id, upload_date,
                             duration, description, source, asr_model, status,
                             error, attempts)
        VALUES (:id, :url, :title, :channel, :channel_id, :upload_date,
                :duration, :description, :source, :asr_model, :status,
                :error, :attempts)
        ON CONFLICT(id) DO UPDATE SET
            url=excluded.url,
            title=COALESCE(excluded.title, videos.title),
            channel=COALESCE(excluded.channel, videos.channel),
            channel_id=COALESCE(excluded.channel_id, videos.channel_id),
            upload_date=COALESCE(excluded.upload_date, videos.upload_date),
            duration=COALESCE(excluded.duration, videos.duration),
            description=COALESCE(excluded.description, videos.description),
            updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        vars(video),
    )
    conn.commit()


def get_video(conn: sqlite3.Connection, video_id: str) -> Video | None:
    row = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    return Video.from_row(row) if row else None


def video_exists(conn: sqlite3.Connection, video_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM videos WHERE id = ?", (video_id,)).fetchone()
    return row is not None


def list_videos(
    conn: sqlite3.Connection,
    status: str | None = None,
    channel_id: str | None = None,
) -> list[Video]:
    query = "SELECT * FROM videos WHERE 1=1"
    params: list[str] = []
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    if channel_id is not None:
        query += " AND channel_id = ?"
        params.append(channel_id)
    query += " ORDER BY created_at ASC"
    rows = conn.execute(query, params).fetchall()
    return [Video.from_row(r) for r in rows]


def set_video_status(
    conn: sqlite3.Connection,
    video_id: str,
    status: str,
    error: str | None = None,
    increment_attempts: bool = False,
) -> None:
    if increment_attempts:
        conn.execute(
            """
            UPDATE videos SET status = ?, error = ?, attempts = attempts + 1,
                               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (status, error, video_id),
        )
    else:
        conn.execute(
            """
            UPDATE videos SET status = ?, error = ?,
                               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (status, error, video_id),
        )
    conn.commit()


def set_video_source(conn: sqlite3.Connection, video_id: str, source: str, asr_model: str | None) -> None:
    conn.execute(
        "UPDATE videos SET source = ?, asr_model = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        (source, asr_model, video_id),
    )
    conn.commit()


# --- Segments ---------------------------------------------------------------

def replace_segments(conn: sqlite3.Connection, video_id: str, segments: Sequence[Segment]) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO segments (video_id, idx, start_s, end_s, text) VALUES (?, ?, ?, ?, ?)",
            [(video_id, s.idx, s.start_s, s.end_s, s.text) for s in segments],
        )


def get_segments(conn: sqlite3.Connection, video_id: str) -> list[Segment]:
    rows = conn.execute(
        "SELECT * FROM segments WHERE video_id = ? ORDER BY idx ASC", (video_id,)
    ).fetchall()
    return [
        Segment(
            id=r["id"], video_id=r["video_id"], idx=r["idx"],
            start_s=r["start_s"], end_s=r["end_s"], text=r["text"],
        )
        for r in rows
    ]


@dataclass
class SearchHit:
    video_id: str
    title: str | None
    channel: str | None
    start_s: float
    end_s: float
    text: str
    snippet: str


def search_fts(
    conn: sqlite3.Connection, query: str, limit: int = 20, channel: str | None = None
) -> list[SearchHit]:
    """Full-text search over segment text using trigram substring matching."""
    if len(query.strip()) < 3:
        raise ValueError("Search query must be at least 3 characters (trigram tokenizer).")

    sql = """
        SELECT v.id AS video_id, v.title AS title, v.channel AS channel,
               s.start_s AS start_s, s.end_s AS end_s, s.text AS text,
               snippet(segments_fts, 0, '[', ']', ' ... ', 10) AS snippet
        FROM segments_fts
        JOIN segments s ON s.id = segments_fts.rowid
        JOIN videos v ON v.id = s.video_id
        WHERE segments_fts MATCH ?
    """
    params: list[str] = [query]
    if channel:
        sql += " AND v.channel = ?"
        params.append(channel)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        SearchHit(
            video_id=r["video_id"], title=r["title"], channel=r["channel"],
            start_s=r["start_s"], end_s=r["end_s"], text=r["text"], snippet=r["snippet"],
        )
        for r in rows
    ]


# --- Chunks / embeddings -----------------------------------------------------

def floats_to_blob(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def blob_to_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def replace_chunks(conn: sqlite3.Connection, video_id: str, chunks: Sequence[Chunk]) -> None:
    with transaction(conn):
        conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
        conn.executemany(
            "INSERT INTO chunks (video_id, start_s, end_s, text, embedding, embed_model) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (video_id, c.start_s, c.end_s, c.text, c.embedding, c.embed_model)
                for c in chunks
            ],
        )


def get_chunks(conn: sqlite3.Connection, video_id: str | None = None) -> list[Chunk]:
    if video_id is not None:
        rows = conn.execute("SELECT * FROM chunks WHERE video_id = ?", (video_id,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
    return [
        Chunk(
            id=r["id"], video_id=r["video_id"], start_s=r["start_s"], end_s=r["end_s"],
            text=r["text"], embedding=r["embedding"], embed_model=r["embed_model"],
        )
        for r in rows
    ]


# --- Summaries ---------------------------------------------------------------

def upsert_summary(conn: sqlite3.Connection, video_id: str, model: str, summary: str) -> None:
    conn.execute(
        """
        INSERT INTO summaries (video_id, model, summary)
        VALUES (?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            model = excluded.model,
            summary = excluded.summary,
            created_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        """,
        (video_id, model, summary),
    )
    conn.commit()


def get_summary(conn: sqlite3.Connection, video_id: str) -> str | None:
    row = conn.execute("SELECT summary FROM summaries WHERE video_id = ?", (video_id,)).fetchone()
    return row["summary"] if row else None
