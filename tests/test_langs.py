from __future__ import annotations

import pytest

from bytx import langs


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("bn", "bn"),
        ("BN", "bn"),
        ("bn-BD", "bn"),
        ("bn_IN", "bn"),
        ("ben", "bn"),
        ("Bangla", "bn"),
        ("bengali", "bn"),
        ("en", "en"),
        ("en-US", "en"),
        ("en-GB", "en"),
        ("eng", "en"),
        ("English", "en"),
        ("  en  ", "en"),
    ],
)
def test_normalize_maps_variants_to_canonical_codes(code, expected):
    assert langs.normalize(code) == expected


@pytest.mark.parametrize("code", [None, "", "hi", "as", "fr", "zh-Hans", "auto", "bnx"])
def test_normalize_rejects_empty_and_unsupported(code):
    assert langs.normalize(code) is None


def test_supported_languages_are_bangla_and_english():
    assert langs.SUPPORTED == ("bn", "en")


def test_display_name():
    assert langs.display_name("bn") == "Bangla"
    assert langs.display_name("en") == "English"
    assert langs.display_name("fr") == "fr"  # unknown codes pass through unchanged
