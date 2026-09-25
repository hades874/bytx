from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bytx import rag, store
from bytx.config import (
    AsrConfig,
    Config,
    FetchConfig,
    OllamaConfig,
    PathsConfig,
    RagConfig,
    SummarizeConfig,
)


class FakeOllama:
    """Duck-types OllamaClient's embed/chat. Embeddings are deterministic
    and controllable per-call via a lookup keyed on the input text.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None, chat_reply: str = "উত্তর"):
        self.vectors = vectors or {}
        self.chat_reply = chat_reply
        self.chat_calls: list[list[dict[str, str]]] = []
        self.embed_calls: list[list[str]] = []

    def embed(self, texts, model=None):
        self.embed_calls.append(list(texts))
        return [self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts]

    def chat(self, messages, model=None, num_ctx=None, temperature=None):
        self.chat_calls.append(messages)
        return self.chat_reply


def make_config() -> Config:
    return Config(
        root=Path("."),
        paths=PathsConfig(
            data_dir=Path("data"), models_dir=Path("models"),
            transcribed_files_dir=Path("yt-transcribed-files"),
        ),
        asr=AsrConfig(
            model="x", device="auto", compute_type="float16", beam_size=5,
            vad_filter=True, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
            repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="", models={},
        ),
        fetch=FetchConfig(force_asr=False, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""),
        ollama=OllamaConfig(host="http://x", embed_model="bge-m3", chat_model="gemma3:4b", num_ctx=8192),
        rag=RagConfig(chunk_seconds=75, chunk_overlap_seconds=15, top_k=3),
        summarize=SummarizeConfig(enabled=True, map_window_minutes=10, language="bn"),
    )


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    store.init_db(c)
    yield c
    c.close()


def make_segments(n: int, seg_len: float = 10.0) -> list[store.Segment]:
    return [
        store.Segment(idx=i, start_s=i * seg_len, end_s=(i + 1) * seg_len, text=f"segment {i}")
        for i in range(n)
    ]


# --- chunk_segments -------------------------------------------------------

def test_chunk_segments_empty():
    assert rag.chunk_segments([], 75, 15) == []


def test_chunk_segments_single_short_segment_becomes_one_chunk():
    segs = make_segments(1, seg_len=5.0)
    chunks = rag.chunk_segments(segs, chunk_seconds=75, overlap_seconds=15)
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0
    assert chunks[0].end_s == 5.0
    assert chunks[0].text == "segment 0"


def test_chunk_segments_splits_into_windows():
    # 20 segments of 10s each = 200s total; chunk_seconds=75 -> multiple chunks
    segs = make_segments(20, seg_len=10.0)
    chunks = rag.chunk_segments(segs, chunk_seconds=75, overlap_seconds=15)
    assert len(chunks) >= 2
    # Chunks should be in chronological order and cover the full range
    assert chunks[0].start_s == 0.0
    assert chunks[-1].end_s == 200.0
    for c in chunks:
        assert c.end_s - c.start_s >= 75 or c is chunks[-1]


def test_chunk_segments_overlap_carries_context_into_next_window():
    segs = make_segments(10, seg_len=10.0)  # 100s total
    chunks = rag.chunk_segments(segs, chunk_seconds=50, overlap_seconds=15)
    assert len(chunks) >= 2
    # second chunk should start before the first chunk's end (i.e. overlap)
    assert chunks[1].start_s < chunks[0].end_s


def test_chunk_segments_invalid_params_raise():
    segs = make_segments(3)
    with pytest.raises(ValueError):
        rag.chunk_segments(segs, chunk_seconds=0, overlap_seconds=0)
    with pytest.raises(ValueError):
        rag.chunk_segments(segs, chunk_seconds=10, overlap_seconds=10)
    with pytest.raises(ValueError):
        rag.chunk_segments(segs, chunk_seconds=10, overlap_seconds=-1)


# --- embed_chunks -----------------------------------------------------------

def test_embed_chunks_attaches_embedding_and_model():
    chunks = [store.Chunk(start_s=0, end_s=10, text="hello", video_id="v1")]
    client = FakeOllama(vectors={"hello": [1.0, 2.0, 3.0]})
    embedded = rag.embed_chunks(client, chunks, "bge-m3")
    assert len(embedded) == 1
    assert embedded[0].embed_model == "bge-m3"
    assert embedded[0].video_id == "v1"
    assert store.blob_to_floats(embedded[0].embedding) == [1.0, 2.0, 3.0]


def test_embed_chunks_empty_list():
    client = FakeOllama()
    assert rag.embed_chunks(client, [], "bge-m3") == []
    assert client.embed_calls == []


# --- index_video --------------------------------------------------------------

def test_index_video_stores_chunks(conn):
    store.upsert_video(conn, store.Video(id="v1", url="https://youtu.be/v1"))
    segs = make_segments(10, seg_len=10.0)
    cfg = make_config()
    client = FakeOllama()
    count = rag.index_video(conn, client, cfg, "v1", segs)
    assert count > 0
    stored = store.get_chunks(conn, "v1")
    assert len(stored) == count
    assert all(c.embed_model == "bge-m3" for c in stored)


# --- semantic_search ------------------------------------------------------

def test_semantic_search_ranks_by_cosine_similarity(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1", title="Video One", channel="Chan A"))
    store.upsert_video(conn, store.Video(id="v2", url="u2", title="Video Two", channel="Chan B"))

    close_vec = [1.0, 0.0, 0.0]
    far_vec = [0.0, 1.0, 0.0]
    store.replace_chunks(conn, "v1", [
        store.Chunk(start_s=0, end_s=10, text="relevant", embedding=store.floats_to_blob(close_vec), embed_model="m"),
    ])
    store.replace_chunks(conn, "v2", [
        store.Chunk(start_s=0, end_s=10, text="irrelevant", embedding=store.floats_to_blob(far_vec), embed_model="m"),
    ])

    cfg = make_config()
    client = FakeOllama(vectors={"query": [1.0, 0.0, 0.0]})
    hits = rag.semantic_search(conn, client, cfg, "query", top_k=2)
    assert len(hits) == 2
    assert hits[0].video_id == "v1"
    assert hits[0].score > hits[1].score
    assert hits[0].title == "Video One"


def test_semantic_search_respects_top_k(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1"))
    chunks = [
        store.Chunk(start_s=i * 10, end_s=i * 10 + 10, text=f"c{i}", embedding=store.floats_to_blob([1.0, float(i), 0.0]), embed_model="m")
        for i in range(5)
    ]
    store.replace_chunks(conn, "v1", chunks)
    cfg = make_config()
    client = FakeOllama(vectors={"q": [1.0, 0.0, 0.0]})
    hits = rag.semantic_search(conn, client, cfg, "q", top_k=2)
    assert len(hits) == 2


def test_semantic_search_filters_by_video_id(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1"))
    store.upsert_video(conn, store.Video(id="v2", url="u2"))
    store.replace_chunks(conn, "v1", [store.Chunk(start_s=0, end_s=1, text="a", embedding=store.floats_to_blob([1, 0, 0]), embed_model="m")])
    store.replace_chunks(conn, "v2", [store.Chunk(start_s=0, end_s=1, text="b", embedding=store.floats_to_blob([1, 0, 0]), embed_model="m")])
    cfg = make_config()
    client = FakeOllama(vectors={"q": [1.0, 0.0, 0.0]})
    hits = rag.semantic_search(conn, client, cfg, "q", video_id="v1")
    assert [h.video_id for h in hits] == ["v1"]


def test_semantic_search_filters_by_channel(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1", channel="A"))
    store.upsert_video(conn, store.Video(id="v2", url="u2", channel="B"))
    store.replace_chunks(conn, "v1", [store.Chunk(start_s=0, end_s=1, text="a", embedding=store.floats_to_blob([1, 0, 0]), embed_model="m")])
    store.replace_chunks(conn, "v2", [store.Chunk(start_s=0, end_s=1, text="b", embedding=store.floats_to_blob([1, 0, 0]), embed_model="m")])
    cfg = make_config()
    client = FakeOllama(vectors={"q": [1.0, 0.0, 0.0]})
    hits = rag.semantic_search(conn, client, cfg, "q", channel="B")
    assert [h.video_id for h in hits] == ["v2"]


def test_semantic_search_no_chunks_returns_empty(conn):
    cfg = make_config()
    client = FakeOllama()
    assert rag.semantic_search(conn, client, cfg, "q") == []


# --- citations / prompt building --------------------------------------------

def test_format_citation_contains_timestamp_and_link():
    hit = rag.SemanticHit(video_id="abc123", title="My Video", channel="C", start_s=125.0, end_s=135.0, text="t", score=0.9)
    citation = rag.format_citation(hit)
    assert "My Video" in citation
    assert "02:05" in citation
    assert "youtu.be/abc123?t=125" in citation


def test_format_citation_falls_back_to_id_when_no_title():
    hit = rag.SemanticHit(video_id="abc123", title=None, channel=None, start_s=0.0, end_s=1.0, text="t", score=0.5)
    citation = rag.format_citation(hit)
    assert "abc123" in citation


def test_build_ask_prompt_includes_context_and_question():
    hit = rag.SemanticHit(video_id="v1", title="T", channel="C", start_s=10.0, end_s=20.0, text="প্রাসঙ্গিক তথ্য", score=0.8)
    messages = rag.build_ask_prompt("এটা কি?", [hit])
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "প্রাসঙ্গিক তথ্য" in messages[1]["content"]
    assert "এটা কি?" in messages[1]["content"]


def test_build_ask_prompt_handles_no_hits():
    messages = rag.build_ask_prompt("প্রশ্ন", [])
    assert "no matching context" in messages[1]["content"]


# --- ask ------------------------------------------------------------------

def test_ask_returns_answer_and_hits(conn):
    store.upsert_video(conn, store.Video(id="v1", url="u1", title="T"))
    store.replace_chunks(conn, "v1", [
        store.Chunk(start_s=0, end_s=10, text="তথ্য", embedding=store.floats_to_blob([1, 0, 0]), embed_model="m"),
    ])
    cfg = make_config()
    client = FakeOllama(vectors={"প্রশ্ন": [1.0, 0.0, 0.0]}, chat_reply="এটাই উত্তর")
    answer, hits = rag.ask(conn, client, cfg, "প্রশ্ন")
    assert answer == "এটাই উত্তর"
    assert len(hits) == 1
    assert len(client.chat_calls) == 1
