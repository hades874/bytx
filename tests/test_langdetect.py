from __future__ import annotations

import pytest

from bytx import langdetect

# Real output of the Bangla model on a Banglish video (Bangla speech, English words).
BANGLISH = (
    "আপনিরা যারা হচ্ছে বাংলাদেশে ব্যাচেলর কম্পিট করেছেন এবং হচ্ছে বিদেশে আসতে যাচ্ছেন "
    "তাদের জন্য এই ভিডিওটা কারণ আমি নিজেও এসেছি এবং আমার অনেক কিছু মনে হয়েছে যেটা আপনাদের বলা দরকার "
)
# Real output of the Bangla model on an English talk: Bengali-script transliteration.
TRANSLITERATED_ENGLISH = (
    "আলার্ভ রিভান্ড বালকাম টু কথার স্লোক আই হোপ ইউ অল আর ডুইং আমেইজিং সো "
    "হোয়েন আই ওয়াজ ইন হাই স্কুল আই ওয়াজ ভেরি ইন্টারেস্টেড ইন হাউ থিংস ওয়ার্ক "
)


def repeat(text: str, times: int = 3) -> str:
    return " ".join([text.strip()] * times)


# --- bangla_word_rate -------------------------------------------------------------

def test_rate_counts_function_words_over_all_words():
    rate, words = langdetect.bangla_word_rate("আমি বাজারে গিয়েছিলাম এবং না")
    assert words == 5
    assert rate == pytest.approx(3 / 5)  # আমি, এবং, না


def test_rate_ignores_punctuation_around_words():
    rate, words = langdetect.bangla_word_rate("আমি, এবং। না? বাজার")
    assert (rate, words) == (pytest.approx(3 / 4), 4)


def test_rate_of_empty_text_is_zero_words():
    assert langdetect.bangla_word_rate("") == (0.0, 0)
    assert langdetect.bangla_word_rate("   ?! ") == (0.0, 0)


def test_banglish_has_far_more_function_words_than_transliterated_english():
    banglish, _ = langdetect.bangla_word_rate(BANGLISH)
    english, _ = langdetect.bangla_word_rate(TRANSLITERATED_ENGLISH)
    assert banglish > 0.25
    assert english < 0.05


def test_latin_text_has_no_bangla_function_words():
    rate, words = langdetect.bangla_word_rate("hello everyone welcome back to the channel")
    assert (rate, words) == (0.0, 7)


# --- classify -----------------------------------------------------------------------

def test_bangla_speech_with_english_words_is_bangla():
    d = langdetect.classify(repeat(BANGLISH), english_below=0.08)
    assert (d.language, d.confident) == ("bn", True)
    assert d.rate > 0.25


def test_transliterated_english_is_english():
    d = langdetect.classify(repeat(TRANSLITERATED_ENGLISH), english_below=0.08)
    assert (d.language, d.confident) == ("en", True)
    assert d.rate < 0.08


def test_english_text_in_latin_script_is_english():
    d = langdetect.classify("so today we are going to talk about the visa process " * 6, english_below=0.08)
    assert d.language == "en"


def test_threshold_decides_the_boundary():
    text = repeat(BANGLISH)
    rate, _ = langdetect.bangla_word_rate(text)
    assert langdetect.classify(text, english_below=rate + 0.01).language == "en"
    assert langdetect.classify(text, english_below=rate).language == "bn"  # not below: stays Bangla


def test_short_sample_falls_back_to_the_default_language():
    d = langdetect.classify(TRANSLITERATED_ENGLISH.split("হোয়েন")[0], english_below=0.08)
    assert d.words < langdetect.MIN_WORDS
    assert (d.language, d.confident) == ("bn", False)


def test_short_sample_uses_the_given_default():
    d = langdetect.classify("just a few words", english_below=0.08, default="en")
    assert (d.language, d.confident) == ("en", False)


def test_empty_sample_is_not_evidence():
    d = langdetect.classify("", english_below=0.08)
    assert (d.language, d.words, d.confident) == ("bn", 0, False)
