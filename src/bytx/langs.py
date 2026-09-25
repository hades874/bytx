"""Supported transcription languages: the one place that knows their codes,
names, and the alternate spellings that YouTube metadata / subtitle tracks use.

Kept dependency-free so config, fetch, and the CLI can all import it.
"""

from __future__ import annotations

BANGLA = "bn"
ENGLISH = "en"
SUPPORTED: tuple[str, ...] = (BANGLA, ENGLISH)

_NAMES = {BANGLA: "Bangla", ENGLISH: "English"}

# Primary subtags and alternate spellings -> canonical code.
_ALIASES = {
    "bn": "bn",
    "ben": "bn",
    "bangla": "bn",
    "bengali": "bn",
    "en": "en",
    "eng": "en",
    "english": "en",
}


def normalize(code: str | None) -> str | None:
    """Maps 'bn-BD', 'en_US', 'ben', 'English'... to 'bn' / 'en'. Returns None
    for empty input or a language we do not support."""
    if not code:
        return None
    primary = code.strip().lower().replace("_", "-").split("-", 1)[0]
    return _ALIASES.get(primary)


def display_name(code: str) -> str:
    """Human-readable name for prompts; unknown codes pass through unchanged."""
    return _NAMES.get(code, code)
