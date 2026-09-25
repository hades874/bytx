"""Bangla-vs-English detection for `bytx add --lang auto`.

Neither of the usual signals works for this library's videos, which are often
Bangla speech full of English words:

- YouTube's language field is the channel default and reads `en-US` on many
  Bangla videos.
- Whisper's language ID scores Bangla at ~0.00 (it says Hindi or English), so
  it cannot tell such videos from real English.

What does separate them is the text the Bangla model writes. Bangla speech,
even with English words mixed in, is full of Bangla grammar words (আছে, করে,
এবং, না, ...). English speech pushed through the same model comes out as
Bengali-script transliteration ("আই হোপ ইউ অল আর ডুইং আমেইজিং") with few of them.
So the detector transcribes a sample of the audio with the Bangla model and
measures that share. Measured on samples of ten clips per video:

- 27 Bangla/Banglish videos: 12.9% - 29.3% (including four vlogs with English
  titles, and ones YouTube tags en-US).
- 5 English videos (TED talks, an IELTS test, an English video that had been
  transcribed as Bangla in this library): 1.5% - 8.5%.

The default cutoff of 10% sits in that gap, a little on the Bangla side, so an
ambiguous video is treated as Bangla (the tool's default) rather than English.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import langs

# Frequent Bangla function words: pronouns, particles, conjunctions, the verbs
# that carry most sentences. Transliterated English seldom produces these.
BANGLA_FUNCTION_WORDS = frozenset(
    """
    এবং আমি আপনি আপনার আপনারা আমার আমরা তারা করে করা করতে করছি করেছেন করেন
    হয় হয়ে হবে হচ্ছে হলো না এই যে তো কি কিন্তু থেকে জন্য একটা একটি আছে তাই যদি
    মানে সাথে যেটা সেটা এটা ওটা কোন কোনো অনেক তখন এখন তাহলে যেহেতু বা ও
    """.split()
)

# Below this many words a rate is noise, so the default language is used.
MIN_WORDS = 40

_PUNCTUATION = ".,?!;:।\"'()[]-"


def bangla_word_rate(text: str) -> tuple[float, int]:
    """(share of words that are Bangla function words, number of words)."""
    words = [w.strip(_PUNCTUATION) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return 0.0, 0
    hits = sum(1 for w in words if w in BANGLA_FUNCTION_WORDS)
    return hits / len(words), len(words)


@dataclass(frozen=True)
class Detection:
    language: str
    rate: float  # share of Bangla function words in the sample
    words: int  # size of the sample
    confident: bool  # False when the sample was too short to decide


def classify(text: str, english_below: float, default: str = langs.BANGLA) -> Detection:
    """English when fewer than `english_below` of the sample's words are Bangla
    function words, Bangla otherwise. A sample under MIN_WORDS words is not
    evidence either way and falls back to `default`."""
    rate, words = bangla_word_rate(text)
    if words < MIN_WORDS:
        return Detection(language=default, rate=rate, words=words, confident=False)
    language = langs.ENGLISH if rate < english_below else langs.BANGLA
    return Detection(language=language, rate=rate, words=words, confident=True)
