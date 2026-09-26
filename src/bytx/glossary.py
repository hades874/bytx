"""Vocabulary list for names, acronyms and jargon that speech recognition
garbles ("Trusted Ireland" for TrustEd Ireland, "ECD UCD" for TCD, UCD).

One plain-text file (`glossary.txt` by default) serves two jobs:

- **Priming**: the canonical terms are handed to Whisper as its initial prompt,
  which biases decoding towards those spellings.
- **Cleaning**: known mis-hearings are replaced with the canonical term, in a
  *copy* of the transcript. The raw ASR text is never modified.

File format, one entry per line (a line starting with `#`, or text after ` #`, is a comment;
blank lines are skipped):

    IRCC                                  a term to prime with
    TrustEd Ireland <= Trusted Ireland    canonical <= variant | variant | ...
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import store

_SEPARATOR = "<="
_COMMENT_RE = re.compile(r"\s#.*$")


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    variants: tuple[str, ...] = ()


def _variant_pattern(variant: str) -> str:
    """Regex source for `variant`, with word boundaries only on
    the sides that end in an ASCII letter/digit. Bangla words take suffixes
    (and combining marks are not \\w), so a boundary there would miss real hits."""
    body = re.escape(variant)
    left = r"(?<![A-Za-z0-9])" if variant[0].isascii() and variant[0].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if variant[-1].isascii() and variant[-1].isalnum() else ""
    return left + body + right


@dataclass
class Glossary:
    entries: list[GlossaryEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        # One pass over the text with every variant as an alternative, longest
        # first, so "Trusted Ireland Scholarship" wins over "Trusted Ireland" and
        # replaced text is never matched again (a canonical term that differs from
        # its variant only in case would otherwise be "fixed" over and over).
        rules = [(v, e.term) for e in self.entries for v in e.variants if v]
        rules.sort(key=lambda r: len(r[0]), reverse=True)
        self._terms_by_variant: dict[str, str] = {}
        for variant, term in rules:
            self._terms_by_variant.setdefault(variant.casefold(), term)
        self._pattern = (
            re.compile("|".join(_variant_pattern(v) for v, _t in rules), re.IGNORECASE)
            if rules
            else None
        )

    def __bool__(self) -> bool:
        return bool(self.entries)

    def terms(self) -> list[str]:
        return [e.term for e in self.entries]

    def prompt(self, max_chars: int) -> str:
        """The canonical terms as a comma-separated prompt of at most `max_chars`
        characters (whole terms only), or "" when nothing fits."""
        picked: list[str] = []
        used = 0
        for term in self.terms():
            extra = len(term) + (2 if picked else 0)
            if used + extra > max_chars:
                break
            picked.append(term)
            used += extra
        return ", ".join(picked)

    def apply(self, text: str) -> tuple[str, int]:
        """`text` with every known variant replaced by its canonical term, and
        how many replacements were made."""
        if self._pattern is None:
            return text, 0
        total = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal total
            term = self._terms_by_variant[match.group(0).casefold()]
            if term != match.group(0):  # already spelled right: not a correction
                total += 1
            return term

        return self._pattern.sub(replace, text), total

    def apply_segments(self, segments: Sequence[store.Segment]) -> tuple[list[store.Segment], int]:
        """Cleaned copies of `segments` (originals untouched) and the total replacements."""
        cleaned: list[store.Segment] = []
        total = 0
        for seg in segments:
            text, n = self.apply(seg.text)
            total += n
            cleaned.append(replace(seg, text=text))
        return cleaned, total


def parse(lines: Iterable[str]) -> Glossary:
    entries: list[GlossaryEntry] = []
    for raw in lines:
        # Only a leading "#" or a "#" after whitespace starts a comment, so "C#" survives.
        line = _COMMENT_RE.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        term, _, rest = line.partition(_SEPARATOR)
        term = term.strip()
        if not term:
            continue
        variants = tuple(v.strip() for v in rest.split("|") if v.strip())
        entries.append(GlossaryEntry(term=term, variants=variants))
    return Glossary(entries)


def load(path: Path | None) -> Glossary:
    """The glossary at `path`; empty when no path is set or the file is missing."""
    if path is None or not path.is_file():
        return Glossary()
    return parse(path.read_text(encoding="utf-8").splitlines())
