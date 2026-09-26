from __future__ import annotations

import pytest

from bytx import compare, store


def seg(idx, start, end, text) -> store.Segment:
    return store.Segment(idx=idx, start_s=start, end_s=end, text=text)


def video() -> store.Video:
    return store.Video(id="abc", url="https://youtu.be/abc", title="Ireland", asr_model="large-v3-turbo")


# --- text helpers -------------------------------------------------------------------

def test_normalize_words_ignores_case_punctuation_and_width():
    assert compare.normalize_words("Hello, WORLD! It's ２০২৬.") == ["hello", "world", "it", "s", "2026"]


def test_normalize_words_keeps_bangla_words_whole():
    assert compare.normalize_words("আমি, বাংলাদেশের ছাত্র।") == ["আমি", "বাংলাদেশের", "ছাত্র"]


def test_extract_numbers_reads_both_scripts_and_thousands_separators():
    assert compare.extract_numbers("fee €14.15, deposit 10,000 and ৩৬০০০, in 2026.") == {
        "14.15", "10000", "36000", "2026",
    }


def test_similarity_bounds():
    assert compare.similarity("the same words", "The same, words!") == 1.0
    assert compare.similarity("alpha beta", "gamma delta") == 0.0
    assert compare.similarity("", "") == 1.0
    assert 0 < compare.similarity("a b c d", "a b x d") < 1


# --- compare ------------------------------------------------------------------------

def test_matching_transcripts_have_no_disagreements():
    ours = [seg(0, 0, 10, "apply to TrustEd Ireland now"), seg(1, 10, 20, "fees are 14.15 euro")]
    ref = [seg(0, 1, 4, "apply to trusted ireland"), seg(1, 4, 9, "now"), seg(2, 11, 19, "fees are 14.15 euro")]

    result = compare.compare(ours, ref)

    assert result.disagreements == []
    assert result.compared == 2
    assert result.mean_similarity == pytest.approx(1.0)
    assert result.missed == []


def test_disagreements_are_worst_first_with_numbers_that_differ():
    ours = [seg(0, 0, 10, "the fee is 36000 euro"), seg(1, 10, 20, "completely different sentence here")]
    ref = [seg(0, 2, 8, "the fee is 36,605 euro"), seg(1, 12, 18, "no overlap at all whatsoever")]

    result = compare.compare(ours, ref, threshold=0.9)

    assert [d.idx for d in result.disagreements] == [1, 0]  # 0% similar before 60%
    first = result.disagreements[1]
    assert first.only_in_transcript == ["36000"]
    assert first.only_in_reference == ["36605"]
    assert result.mean_similarity < 0.5


def test_threshold_decides_what_counts_as_a_disagreement():
    ours = [seg(0, 0, 10, "one two three four")]
    ref = [seg(0, 2, 8, "one two three five")]  # 75% similar

    assert compare.compare(ours, ref, threshold=0.5).disagreements == []
    assert len(compare.compare(ours, ref, threshold=0.8).disagreements) == 1


def test_segments_without_reference_text_are_not_scored_as_wrong():
    ours = [seg(0, 0, 10, "some words"), seg(1, 10, 20, "more words")]
    ref = [seg(0, 2, 8, "some words")]

    result = compare.compare(ours, ref)

    assert (result.compared, result.no_reference) == (1, 1)
    assert result.disagreements == []


def test_reference_speech_outside_every_segment_is_reported_as_missed():
    ours = [seg(0, 0, 10, "we cover the start")]
    ref = [
        seg(0, 2, 8, "we cover the start"),
        seg(1, 30, 33, "this part the transcript"),  # a run of two cues, merged
        seg(2, 33.5, 36, "never picked up at all"),
        seg(3, 90, 91, "uh huh"),  # too short to matter
    ]

    result = compare.compare(ours, ref)

    assert len(result.missed) == 1
    m = result.missed[0]
    assert (m.start_s, m.end_s) == (30, 36)
    assert "never picked up" in m.text


def test_a_cue_is_assigned_to_the_segment_holding_its_midpoint():
    ours = [seg(0, 0, 10, "first half"), seg(1, 10, 20, "second half")]
    ref = [seg(0, 8, 13, "second half")]  # midpoint 10.5 -> segment 1

    result = compare.compare(ours, ref)

    assert result.compared == 1 and result.no_reference == 1
    assert result.disagreements == []


def test_nothing_to_compare():
    result = compare.compare([seg(0, 0, 10, "words here")], [])
    assert result.mean_similarity is None and result.compared == 0
    assert "Nothing to compare" in compare.render_markdown(video(), result, "YouTube auto-captions", 0.5)


def test_render_markdown_lists_disagreements_and_missed_speech():
    ours = [seg(0, 0, 10, "the fee is 36000 euro")]
    ref = [seg(0, 2, 8, "the price was 36,605 euros"), seg(1, 40, 46, "a stretch we never transcribed")]
    result = compare.compare(ours, ref, threshold=0.9)

    text = compare.render_markdown(video(), result, "YouTube auto-captions", 0.9)

    assert text.startswith("# Comparison: Ireland")
    assert "large-v3-turbo" in text
    assert "0:00-0:10" in text
    assert "Transcript: the fee is 36000 euro" in text
    assert "YouTube auto-captions: the price was 36,605 euros" in text
    assert "only in YouTube auto-captions: 36605" in text
    assert "0:40-0:46: a stretch we never transcribed" in text
