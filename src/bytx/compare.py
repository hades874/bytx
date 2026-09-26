"""Cross-checks a transcript against an independent second transcript of the
same video (in practice YouTube's auto-generated captions).

Two different recognisers rarely make the same mistake, so the places where
they disagree are where an error is most likely: a wrong name, a garbled
sentence, a number heard differently. It also finds speech the second
transcript has and ours lacks, which is how dropped stretches show up.

A disagreement marks a spot to listen to, not a proven error: either side may
be the wrong one, and YouTube's Bangla captions in particular are weak.
Numbers are compared as digits, so "ten thousand" against "10,000" will show
up as a difference even though both are right.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from . import store
from .quality import format_time

DEFAULT_THRESHOLD = 0.5
# Reference-only speech shorter than this many words is noise, not a dropped stretch.
_MIN_MISSED_WORDS = 4

_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_NOT_WORD_RE = re.compile(r"[^\wঀ-৿]+")  # \w misses Bangla vowel signs


@dataclass
class Disagreement:
    idx: int
    start_s: float
    end_s: float
    similarity: float
    text: str  # our transcript
    ref_text: str
    only_in_transcript: list[str] = field(default_factory=list)  # numbers
    only_in_reference: list[str] = field(default_factory=list)


@dataclass
class MissedSpeech:
    start_s: float
    end_s: float
    text: str


@dataclass
class Comparison:
    mean_similarity: float | None  # over the segments that had reference text
    compared: int
    no_reference: int  # segments with no reference text to compare against
    disagreements: list[Disagreement] = field(default_factory=list)  # worst first
    missed: list[MissedSpeech] = field(default_factory=list)  # in reference, not in transcript


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold().translate(_BANGLA_DIGITS)


def normalize_words(text: str) -> list[str]:
    """Lower-cased words with punctuation dropped, for fair comparison."""
    return _NOT_WORD_RE.sub(" ", _clean(text)).split()


def extract_numbers(text: str) -> set[str]:
    """The numbers in `text` as plain digits ("10,000" -> "10000"), with
    Bangla digits converted."""
    return {n.replace(",", "") for n in _NUMBER_RE.findall(_clean(text))}


def similarity(a: str, b: str) -> float:
    """Word-level similarity from 0 (nothing shared) to 1 (identical)."""
    words_a, words_b = normalize_words(a), normalize_words(b)
    if not words_a and not words_b:
        return 1.0
    return SequenceMatcher(None, words_a, words_b, autojunk=False).ratio()


def _midpoint(seg: store.Segment) -> float:
    return (seg.start_s + seg.end_s) / 2


def compare(
    segments: Sequence[store.Segment],
    reference: Sequence[store.Segment],
    threshold: float = DEFAULT_THRESHOLD,
) -> Comparison:
    """Compares each of `segments` with the reference cues whose midpoint falls
    inside it. Segments scoring below `threshold` are reported, worst first,
    along with reference speech that lies outside every segment."""
    ordered = sorted(segments, key=lambda s: s.start_s)
    ref_by_segment: list[list[store.Segment]] = [[] for _ in ordered]
    unmatched: list[store.Segment] = []
    for cue in sorted(reference, key=lambda c: c.start_s):
        mid = _midpoint(cue)
        for i, seg in enumerate(ordered):
            if seg.start_s <= mid <= seg.end_s:
                ref_by_segment[i].append(cue)
                break
        else:
            unmatched.append(cue)

    scores: list[float] = []
    disagreements: list[Disagreement] = []
    no_reference = 0
    for seg, cues in zip(ordered, ref_by_segment, strict=True):
        ref_text = " ".join(c.text for c in cues)
        if not normalize_words(ref_text):
            no_reference += 1
            continue
        score = similarity(seg.text, ref_text)
        scores.append(score)
        if score < threshold:
            ours, theirs = extract_numbers(seg.text), extract_numbers(ref_text)
            disagreements.append(Disagreement(
                idx=seg.idx, start_s=seg.start_s, end_s=seg.end_s, similarity=score,
                text=seg.text, ref_text=ref_text,
                only_in_transcript=sorted(ours - theirs), only_in_reference=sorted(theirs - ours),
            ))
    disagreements.sort(key=lambda d: d.similarity)

    return Comparison(
        mean_similarity=sum(scores) / len(scores) if scores else None,
        compared=len(scores),
        no_reference=no_reference,
        disagreements=disagreements,
        missed=_merge_missed(unmatched),
    )


def _merge_missed(cues: list[store.Segment]) -> list[MissedSpeech]:
    """Runs of neighbouring unmatched reference cues, as single stretches; runs
    with only a word or two are dropped."""
    runs: list[list[store.Segment]] = []
    for cue in cues:
        if runs and cue.start_s - runs[-1][-1].end_s <= 2.0:
            runs[-1].append(cue)
        else:
            runs.append([cue])
    missed = []
    for run in runs:
        text = " ".join(c.text for c in run)
        if len(normalize_words(text)) >= _MIN_MISSED_WORDS:
            missed.append(MissedSpeech(run[0].start_s, max(c.end_s for c in run), text))
    return missed


def render_markdown(
    video: store.Video, comparison: Comparison, reference_name: str, threshold: float
) -> str:
    c = comparison
    lines = [
        f"# Comparison: {video.title or video.id}",
        "",
        f"{video.url}",
        "",
        f"Transcript ({video.asr_model or video.source or 'unknown'}) against {reference_name}.",
        "A disagreement marks a spot to listen to, not a proven error: either side can be wrong.",
        "",
    ]
    if c.mean_similarity is None:
        lines.append("Nothing to compare: the reference has no text inside any transcript segment.")
    else:
        lines.append(
            f"- Mean similarity: **{c.mean_similarity:.0%}** over {c.compared} segments "
            f"({c.no_reference} had no reference text)"
        )
        lines.append(f"- Segments below {threshold:.0%}: **{len(c.disagreements)}**")
    lines.append(f"- Reference speech missing from the transcript: **{len(c.missed)}** stretch(es)")

    if c.disagreements:
        lines += ["", "## Disagreements (worst first)", ""]
        for d in c.disagreements:
            lines.append(f"### {format_time(d.start_s)}-{format_time(d.end_s)} · {d.similarity:.0%} similar")
            lines.append(f"- Transcript: {d.text}")
            lines.append(f"- {reference_name}: {d.ref_text}")
            if d.only_in_transcript or d.only_in_reference:
                lines.append(
                    f"- Numbers only in transcript: {', '.join(d.only_in_transcript) or 'none'}; "
                    f"only in {reference_name}: {', '.join(d.only_in_reference) or 'none'}"
                )
            lines.append("")
    if c.missed:
        lines += ["", "## In the reference but not in the transcript", ""]
        for m in c.missed:
            lines.append(f"- {format_time(m.start_s)}-{format_time(m.end_s)}: {m.text}")
    return "\n".join(lines).rstrip("\n") + "\n"
