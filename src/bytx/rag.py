"""Chunking, embedding, and retrieval for semantic search and `ask`.

Embeddings are stored as float32 BLOBs in SQLite; retrieval is brute-force
cosine similarity over numpy arrays. This is fast enough for a personal
library of up to a few thousand videos and needs no extra vector DB.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import store
from .config import Config
from .llm import OllamaClient


def chunk_segments(
    segments: Sequence[store.Segment], chunk_seconds: int, overlap_seconds: int
) -> list[store.Chunk]:
    """Greedily merges consecutive segments into ~chunk_seconds windows with
    overlap_seconds of trailing overlap carried into the next window, so a
    fact near a window boundary isn't split away from its context.
    """
    if not segments:
        return []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_seconds")

    chunks: list[store.Chunk] = []
    window: list[store.Segment] = []

    def flush() -> None:
        if not window:
            return
        text = " ".join(s.text for s in window)
        chunks.append(store.Chunk(start_s=window[0].start_s, end_s=window[-1].end_s, text=text))

    for seg in segments:
        if not window:
            window.append(seg)
            continue
        window.append(seg)
        if seg.end_s - window[0].start_s >= chunk_seconds:
            flush()
            overlap_from = seg.end_s - overlap_seconds
            window = [s for s in window if s.end_s > overlap_from]

    flush()
    return chunks


def embed_chunks(client: OllamaClient, chunks: Sequence[store.Chunk], embed_model: str) -> list[store.Chunk]:
    if not chunks:
        return []
    vectors = client.embed([c.text for c in chunks], model=embed_model)
    out = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        out.append(
            store.Chunk(
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                text=chunk.text,
                video_id=chunk.video_id,
                embedding=store.floats_to_blob(vec),
                embed_model=embed_model,
            )
        )
    return out


def index_video(
    conn, client: OllamaClient, cfg: Config, video_id: str, segments: Sequence[store.Segment]
) -> int:
    """Chunks a video's segments, embeds them, and stores the result,
    replacing any previous chunks for that video. Returns the chunk count.
    """
    chunks = chunk_segments(segments, cfg.rag.chunk_seconds, cfg.rag.chunk_overlap_seconds)
    embedded = embed_chunks(client, chunks, cfg.ollama.embed_model)
    store.replace_chunks(conn, video_id, embedded)
    return len(embedded)


@dataclass
class SemanticHit:
    video_id: str
    title: str | None
    channel: str | None
    start_s: float
    end_s: float
    text: str
    score: float


def _cosine_sim_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query / (np.linalg.norm(query) + 1e-9)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix_normed = matrix / matrix_norms
    return matrix_normed @ query_norm


def semantic_search(
    conn,
    client: OllamaClient,
    cfg: Config,
    query: str,
    top_k: int | None = None,
    video_id: str | None = None,
    channel: str | None = None,
) -> list[SemanticHit]:
    chunks = store.get_chunks(conn, video_id=video_id)
    chunks = [c for c in chunks if c.embedding is not None]
    if not chunks:
        return []

    query_vec = np.array(client.embed([query], model=cfg.ollama.embed_model)[0], dtype=np.float32)
    matrix = np.array([store.blob_to_floats(c.embedding) for c in chunks], dtype=np.float32)
    scores = _cosine_sim_matrix(query_vec, matrix)

    video_cache: dict[str, store.Video | None] = {}

    def get_video_cached(vid: str) -> store.Video | None:
        if vid not in video_cache:
            video_cache[vid] = store.get_video(conn, vid)
        return video_cache[vid]

    order = np.argsort(-scores)
    k = top_k if top_k is not None else cfg.rag.top_k

    hits: list[SemanticHit] = []
    for i in order:
        chunk = chunks[i]
        video = get_video_cached(chunk.video_id)
        if channel is not None and (video is None or video.channel != channel):
            continue
        hits.append(
            SemanticHit(
                video_id=chunk.video_id,
                title=video.title if video else None,
                channel=video.channel if video else None,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                text=chunk.text,
                score=float(scores[i]),
            )
        )
        if len(hits) >= k:
            break
    return hits


def format_citation(hit: SemanticHit) -> str:
    minutes, seconds = divmod(int(hit.start_s), 60)
    title = hit.title or hit.video_id
    return f"[{title} @ {minutes:02d}:{seconds:02d}] https://youtu.be/{hit.video_id}?t={int(hit.start_s)}"


ASK_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the contents of "
    "YouTube video transcripts. Answer ONLY using the provided context "
    "excerpts. If the context does not contain the answer, say so plainly "
    "instead of guessing. Answer in Bangla, and after your answer, list the "
    "citations you used exactly as given, one per line."
)


def build_ask_prompt(question: str, hits: Sequence[SemanticHit]) -> list[dict[str, str]]:
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(f"[{i}] {format_citation(hit)}\n{hit.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(no matching context found)"
    user_content = f"Context excerpts:\n\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def ask(
    conn,
    client: OllamaClient,
    cfg: Config,
    question: str,
    video_id: str | None = None,
) -> tuple[str, list[SemanticHit]]:
    hits = semantic_search(conn, client, cfg, question, video_id=video_id)
    messages = build_ask_prompt(question, hits)
    answer = client.chat(messages, model=cfg.ollama.chat_model, num_ctx=cfg.ollama.num_ctx)
    return answer, hits
