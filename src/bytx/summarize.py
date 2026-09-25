"""Map-reduce summarization of a video's transcript via the local Ollama
chat model: summarize each ~map_window_minutes window (map), then combine
those partial summaries into one final summary (reduce).
"""

from __future__ import annotations

from collections.abc import Sequence

from . import langs, store
from .config import Config
from .llm import OllamaClient


def _language_instruction(language: str) -> str:
    return f"Write your response in {langs.display_name(language)}."


def summary_language(cfg: Config, video: store.Video) -> str:
    """`summarize.language`, or the video's own language when it is "auto"."""
    if cfg.summarize.language != "auto":
        return cfg.summarize.language
    return video.language or langs.BANGLA


def window_segments(
    segments: Sequence[store.Segment], window_minutes: int
) -> list[list[store.Segment]]:
    """Splits segments into consecutive windows of ~window_minutes each,
    without splitting a segment across windows.
    """
    if not segments:
        return []
    window_seconds = window_minutes * 60
    windows: list[list[store.Segment]] = []
    current: list[store.Segment] = []
    for seg in segments:
        if current and seg.start_s - current[0].start_s >= window_seconds:
            windows.append(current)
            current = []
        current.append(seg)
    if current:
        windows.append(current)
    return windows


def _map_summarize(client: OllamaClient, cfg: Config, text: str, language: str) -> str:
    prompt = (
        "Summarize the following excerpt from a video transcript in a few "
        "concise sentences, keeping key facts, names, and numbers. "
        f"{_language_instruction(language)}\n\n"
        f"Transcript excerpt:\n{text}"
    )
    return client.chat(
        [{"role": "user", "content": prompt}],
        model=cfg.ollama.chat_model,
        num_ctx=cfg.ollama.num_ctx,
    )


def _reduce_summarize(
    client: OllamaClient, cfg: Config, title: str | None, partial_summaries: list[str], language: str
) -> str:
    if len(partial_summaries) == 1:
        joined = partial_summaries[0]
    else:
        joined = "\n\n".join(f"({i + 1}) {s}" for i, s in enumerate(partial_summaries))

    title_line = f'Video title: "{title}"\n\n' if title else ""
    prompt = (
        f"{title_line}Below are summaries of consecutive parts of a video's "
        "transcript, in order. Combine them into a single coherent summary "
        "of the whole video: a short overview paragraph followed by a few "
        f"bullet points of key takeaways. {_language_instruction(language)}\n\n"
        f"Part summaries:\n{joined}"
    )
    return client.chat(
        [{"role": "user", "content": prompt}],
        model=cfg.ollama.chat_model,
        num_ctx=cfg.ollama.num_ctx,
    )


def summarize_video(
    client: OllamaClient,
    cfg: Config,
    video: store.Video,
    segments: Sequence[store.Segment],
) -> str:
    """Map-reduce summary of a full transcript. Falls back to a single
    direct summarization pass if the transcript fits in one window.
    """
    windows = window_segments(segments, cfg.summarize.map_window_minutes)
    if not windows:
        return ""
    language = summary_language(cfg, video)

    if len(windows) == 1:
        text = " ".join(s.text for s in windows[0])
        return _map_summarize(client, cfg, text, language)

    partial_summaries = [
        _map_summarize(client, cfg, " ".join(s.text for s in window), language)
        for window in windows
    ]
    return _reduce_summarize(client, cfg, video.title, partial_summaries, language)

