"""Measures how far a transcript can be trusted, from the stored segments and
the video's duration alone (so it also works for transcripts made before
per-clip confidence was recorded).

Whisper output is good for leads but can silently drop stretches of speech or
make up text, and nothing in a plain transcript shows where. The signals here:

- words per minute: normal speech runs about 100-200; far below means speech
  was skipped, far above means repeated or invented text;
- coverage and gaps: how much of the video the segments span, and any long
  stretch (including the start and end) with no text at all;
- low-confidence clips: the model's own log-probability / no-speech scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from . import store
from .config import QualityConfig


@dataclass
class Gap:
    start_s: float
    end_s: float

    @property
    def seconds(self) -> float:
        return self.end_s - self.start_s


@dataclass
class Quality:
    words: int
    duration_s: float | None
    words_per_minute: float | None
    coverage: float | None  # share of the video's duration covered by segments
    gaps: list[Gap] = field(default_factory=list)
    segments: int = 0
    low_confidence: list[int] = field(default_factory=list)  # segment idx values
    confidence_known: bool = False  # any segment carries ASR confidence
    mean_avg_logprob: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def suspect(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["gaps"] = [
            {"start": g.start_s, "end": g.end_s, "seconds": round(g.seconds, 1)} for g in self.gaps
        ]
        for key in ("words_per_minute", "coverage", "mean_avg_logprob"):
            if data[key] is not None:
                data[key] = round(data[key], 3)
        return data


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _find_gaps(
    segments: Sequence[store.Segment], duration_s: float | None, min_gap_s: float
) -> list[Gap]:
    gaps: list[Gap] = []
    cursor = 0.0
    for seg in segments:
        if seg.start_s - cursor > min_gap_s:
            gaps.append(Gap(cursor, seg.start_s))
        cursor = max(cursor, seg.end_s)
    if duration_s is not None and duration_s - cursor > min_gap_s:
        gaps.append(Gap(cursor, duration_s))
    return gaps


def assess(
    video: store.Video, segments: Sequence[store.Segment], cfg: QualityConfig | None = None
) -> Quality:
    cfg = cfg or QualityConfig()
    ordered = sorted(segments, key=lambda s: s.start_s)
    duration = video.duration if video.duration and video.duration > 0 else None
    words = sum(len(s.text.split()) for s in ordered)

    wpm = words / (duration / 60) if duration else None
    covered = sum(max(0.0, min(s.end_s, duration or s.end_s) - s.start_s) for s in ordered)
    coverage = min(1.0, covered / duration) if duration else None
    gaps = _find_gaps(ordered, duration, cfg.max_gap_seconds)

    scored = [s for s in ordered if s.avg_logprob is not None]
    low = [
        s.idx
        for s in ordered
        if (s.avg_logprob is not None and s.avg_logprob < cfg.min_avg_logprob)
        or (s.no_speech_prob is not None and s.no_speech_prob > cfg.max_no_speech_prob)
    ]

    quality = Quality(
        words=words,
        duration_s=duration,
        words_per_minute=wpm,
        coverage=coverage,
        gaps=gaps,
        segments=len(ordered),
        low_confidence=low,
        confidence_known=bool(scored),
        mean_avg_logprob=(sum(s.avg_logprob for s in scored) / len(scored)) if scored else None,
    )
    quality.flags = _flags(quality, cfg)
    return quality


def _flags(q: Quality, cfg: QualityConfig) -> list[str]:
    if q.segments == 0:
        return ["no transcript text"]
    flags: list[str] = []
    if q.words_per_minute is not None:
        if q.words_per_minute < cfg.min_wpm:
            flags.append(
                f"only {q.words_per_minute:.0f} words/min (expected {cfg.min_wpm:.0f}+): "
                "speech may be missing"
            )
        elif q.words_per_minute > cfg.max_wpm:
            flags.append(
                f"{q.words_per_minute:.0f} words/min (expected under {cfg.max_wpm:.0f}): "
                "text may be repeated or invented"
            )
    if q.gaps:
        longest = max(q.gaps, key=lambda g: g.seconds)
        flags.append(
            f"{len(q.gaps)} gap(s) over {cfg.max_gap_seconds:.0f}s with no text "
            f"(longest {longest.seconds:.0f}s at {format_time(longest.start_s)})"
        )
    if q.low_confidence:
        flags.append(f"{len(q.low_confidence)} of {q.segments} clips are low-confidence")
    return flags
