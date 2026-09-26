from __future__ import annotations

from bytx import glossary, store

SAMPLE = """
# names the ASR garbles
IRCC
TrustEd Ireland <= Trusted Ireland | Trust Ed Ireland   # trailing comment
TCD <= ECD | TCU
C#
"""


def make() -> glossary.Glossary:
    return glossary.parse(SAMPLE.splitlines())


def test_parse_reads_terms_variants_and_skips_comments_and_blanks():
    g = make()
    assert g.terms() == ["IRCC", "TrustEd Ireland", "TCD", "C#"]
    assert g.entries[1].variants == ("Trusted Ireland", "Trust Ed Ireland")


def test_a_hash_inside_a_term_is_not_a_comment():
    assert "C#" in make().terms()


def test_apply_replaces_variants_case_insensitively_and_counts():
    text, n = make().apply("I applied to trusted ireland and Trust Ed Ireland.")
    assert text == "I applied to TrustEd Ireland and TrustEd Ireland."
    assert n == 2


def test_text_already_spelled_right_is_not_counted_or_changed():
    assert make().apply("TrustEd Ireland and TCD") == ("TrustEd Ireland and TCD", 0)


def test_apply_respects_word_boundaries_for_latin_variants():
    text, n = make().apply("the ECDC report, not ECD.")
    assert text == "the ECDC report, not TCD."
    assert n == 1


def test_apply_prefers_the_longest_variant():
    g = glossary.parse(["Big Uni <= Trusted Ireland Scholarship", "TrustEd Ireland <= Trusted Ireland"])
    text, _ = g.apply("Trusted Ireland Scholarship and Trusted Ireland")
    assert text == "Big Uni and TrustEd Ireland"


def test_bangla_variants_match_inside_suffixed_words():
    g = glossary.parse(["আয়ারল্যান্ড <= আয়ারল্যান্ট"])
    text, n = g.apply("আয়ারল্যান্টের ভিসা")
    assert text == "আয়ারল্যান্ডের ভিসা"
    assert n == 1


def test_apply_with_no_matches_returns_the_text_unchanged():
    assert make().apply("nothing to fix") == ("nothing to fix", 0)


def test_apply_segments_returns_copies_and_never_touches_the_originals():
    segs = [
        store.Segment(idx=0, start_s=0, end_s=2, text="Trusted Ireland", avg_logprob=-0.2),
        store.Segment(idx=1, start_s=2, end_s=4, text="fine"),
    ]
    cleaned, n = make().apply_segments(segs)

    assert n == 1
    assert [s.text for s in cleaned] == ["TrustEd Ireland", "fine"]
    assert [s.text for s in segs] == ["Trusted Ireland", "fine"]  # raw text preserved
    assert cleaned[0].avg_logprob == -0.2 and cleaned[0].start_s == 0  # everything else carried over


def test_prompt_lists_whole_terms_within_the_budget():
    g = make()
    assert g.prompt(1000) == "IRCC, TrustEd Ireland, TCD, C#"
    assert g.prompt(21) == "IRCC, TrustEd Ireland"  # ", TCD" would overflow it
    assert g.prompt(3) == ""


def test_an_empty_glossary_is_falsey_and_harmless():
    g = glossary.Glossary()
    assert not g
    assert g.prompt(100) == ""
    assert g.apply("x") == ("x", 0)


def test_load_handles_missing_file_and_none(tmp_path):
    assert not glossary.load(None)
    assert not glossary.load(tmp_path / "missing.txt")


def test_load_reads_a_utf8_file(tmp_path):
    path = tmp_path / "glossary.txt"
    path.write_text("আয়ারল্যান্ড\nTCD <= ECD\n", encoding="utf-8")
    g = glossary.load(path)
    assert g.terms() == ["আয়ারল্যান্ড", "TCD"]
