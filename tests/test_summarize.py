from __future__ import annotations

from pathlib import Path

from bytx import store, summarize
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
    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies) if replies else ["সারাংশ"]
        self.chat_calls: list[str] = []

    def chat(self, messages, model=None, num_ctx=None, temperature=None):
        self.chat_calls.append(messages[-1]["content"])
        if len(self.replies) == 1:
            return self.replies[0]
        return self.replies.pop(0)


def make_config(language: str = "bn", map_window_minutes: int = 10) -> Config:
    return Config(
        root=Path("."),
        paths=PathsConfig(data_dir=Path("data"), models_dir=Path("models")),
        asr=AsrConfig(
            model="x", device="auto", compute_type="float16", beam_size=5,
            vad_filter=True, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
            repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="", models={},
        ),
        fetch=FetchConfig(force_asr=False, sleep_interval_s=0, max_sleep_interval_s=0, cookies_from_browser=""),
        ollama=OllamaConfig(host="http://x", embed_model="bge-m3", chat_model="gemma3:4b", num_ctx=8192),
        rag=RagConfig(chunk_seconds=75, chunk_overlap_seconds=15, top_k=8),
        summarize=SummarizeConfig(enabled=True, map_window_minutes=map_window_minutes, language=language),
    )


def make_segments(n: int, seg_len: float = 60.0) -> list[store.Segment]:
    return [
        store.Segment(idx=i, start_s=i * seg_len, end_s=(i + 1) * seg_len, text=f"অংশ {i}")
        for i in range(n)
    ]


# --- window_segments ------------------------------------------------------

def test_window_segments_empty():
    assert summarize.window_segments([], 10) == []


def test_window_segments_single_window_when_short():
    segs = make_segments(5, seg_len=60.0)  # 5 minutes total
    windows = summarize.window_segments(segs, window_minutes=10)
    assert len(windows) == 1
    assert len(windows[0]) == 5


def test_window_segments_splits_at_boundary():
    # 20 segments * 60s = 20 minutes; window_minutes=10 -> 2 windows
    segs = make_segments(20, seg_len=60.0)
    windows = summarize.window_segments(segs, window_minutes=10)
    assert len(windows) == 2
    assert windows[0][0].start_s == 0.0
    # all segments accounted for, none split
    total = sum(len(w) for w in windows)
    assert total == 20


def test_window_segments_does_not_split_a_segment():
    segs = make_segments(3, seg_len=60.0)
    windows = summarize.window_segments(segs, window_minutes=1)
    # each segment is 60s = 1 window_minutes, so likely 3 separate windows,
    # but the important invariant is every original segment appears exactly once
    all_segs = [s for w in windows for s in w]
    assert len(all_segs) == 3


# --- summarize_video (map-only path) ----------------------------------------

def test_summarize_video_single_window_calls_chat_once():
    segs = make_segments(3, seg_len=60.0)  # 3 minutes, fits in one 10-min window
    cfg = make_config(map_window_minutes=10)
    client = FakeOllama(replies=["একক সারাংশ"])
    video = store.Video(id="v1", url="u1", title="শিরোনাম")

    result = summarize.summarize_video(client, cfg, video, segs)
    assert result == "একক সারাংশ"
    assert len(client.chat_calls) == 1
    assert "Bangla" in client.chat_calls[0]


def test_summarize_video_empty_segments_returns_empty_string():
    cfg = make_config()
    client = FakeOllama()
    video = store.Video(id="v1", url="u1")
    result = summarize.summarize_video(client, cfg, video, [])
    assert result == ""
    assert client.chat_calls == []


# --- summarize_video (map-reduce path) ---------------------------------------

def test_summarize_video_multi_window_does_map_then_reduce():
    segs = make_segments(20, seg_len=60.0)  # 20 minutes -> 2 windows @ 10 min
    cfg = make_config(map_window_minutes=10)
    client = FakeOllama(replies=["অংশ সারাংশ ১", "অংশ সারাংশ ২", "চূড়ান্ত সারাংশ"])
    video = store.Video(id="v1", url="u1", title="ভিডিও শিরোনাম")

    result = summarize.summarize_video(client, cfg, video, segs)

    assert result == "চূড়ান্ত সারাংশ"
    assert len(client.chat_calls) == 3  # 2 map calls + 1 reduce call
    # The reduce call should reference both partial summaries and the title
    reduce_prompt = client.chat_calls[-1]
    assert "অংশ সারাংশ ১" in reduce_prompt
    assert "অংশ সারাংশ ২" in reduce_prompt
    assert "ভিডিও শিরোনাম" in reduce_prompt


def test_summarize_video_reduce_omits_title_line_when_missing():
    segs = make_segments(20, seg_len=60.0)
    cfg = make_config(map_window_minutes=10)
    client = FakeOllama(replies=["s1", "s2", "final"])
    video = store.Video(id="v1", url="u1", title=None)

    summarize.summarize_video(client, cfg, video, segs)
    reduce_prompt = client.chat_calls[-1]
    assert "Video title" not in reduce_prompt


def test_summarize_video_english_language_instruction():
    segs = make_segments(3, seg_len=60.0)
    cfg = make_config(language="en")
    client = FakeOllama(replies=["summary"])
    video = store.Video(id="v1", url="u1")
    summarize.summarize_video(client, cfg, video, segs)
    assert "English" in client.chat_calls[0]
