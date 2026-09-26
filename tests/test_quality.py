from __future__ import annotations

from bytx import quality, store
from bytx.config import QualityConfig


def video(duration: float | None = 60.0) -> store.Video:
    return store.Video(id="v", url="u", duration=duration)


def seg(idx, start, end, words=10, **kw) -> store.Segment:
    return store.Segment(idx=idx, start_s=start, end_s=end, text=" ".join(["w"] * words), **kw)


def healthy() -> list[store.Segment]:
    # 130 words over a 60 s video, fully covered.
    return [seg(0, 0, 30, 65), seg(1, 30, 60, 65)]


def test_healthy_transcript_has_no_flags():
    q = quality.assess(video(), healthy())
    assert q.flags == []
    assert not q.suspect
    assert q.words == 130
    assert q.words_per_minute == 130
    assert q.coverage == 1.0
    assert q.gaps == []


def test_low_words_per_minute_is_flagged():
    q = quality.assess(video(), [seg(0, 0, 60, 30)])
    assert any("30 words/min" in f and "missing" in f for f in q.flags)


def test_high_words_per_minute_is_flagged():
    q = quality.assess(video(), [seg(0, 0, 60, 400)])
    assert any("400 words/min" in f and "repeated or invented" in f for f in q.flags)


def test_thresholds_come_from_config():
    cfg = QualityConfig(min_wpm=10, max_wpm=50)
    assert quality.assess(video(), [seg(0, 0, 60, 30)], cfg).flags == []
    assert quality.assess(video(), [seg(0, 0, 60, 100)], cfg).flags != []


def test_a_gap_between_segments_is_found_and_reported():
    segs = [seg(0, 0, 10, 100), seg(1, 40, 60, 100)]
    q = quality.assess(video(), segs)
    assert [(g.start_s, g.end_s) for g in q.gaps] == [(10, 40)]
    assert any("1 gap(s) over 10s" in f and "longest 30s at 0:10" in f for f in q.flags)
    assert q.coverage == 0.5


def test_head_and_tail_gaps_count():
    q = quality.assess(video(120.0), [seg(0, 20, 60, 400)])
    assert [(g.start_s, g.end_s) for g in q.gaps] == [(0.0, 20), (60, 120.0)]


def test_short_pauses_are_not_gaps():
    q = quality.assess(video(), [seg(0, 0, 25, 65), seg(1, 30, 55, 65)])
    assert q.gaps == []


def test_segments_are_ordered_by_time_before_measuring():
    q = quality.assess(video(), list(reversed(healthy())))
    assert q.gaps == [] and q.flags == []


def test_low_confidence_clips_are_listed_and_flagged():
    segs = [
        seg(0, 0, 20, 45, avg_logprob=-0.3, no_speech_prob=0.01),
        seg(1, 20, 40, 45, avg_logprob=-1.4, no_speech_prob=0.01),
        seg(2, 40, 60, 45, avg_logprob=-0.3, no_speech_prob=0.9),
    ]
    q = quality.assess(video(), segs)
    assert q.low_confidence == [1, 2]
    assert q.confidence_known
    assert any("2 of 3 clips are low-confidence" in f for f in q.flags)
    assert round(q.mean_avg_logprob, 3) == round((-0.3 - 1.4 - 0.3) / 3, 3)


def test_without_confidence_nothing_is_low_confidence():
    q = quality.assess(video(), healthy())
    assert q.low_confidence == [] and not q.confidence_known and q.mean_avg_logprob is None


def test_unknown_duration_skips_duration_based_checks():
    q = quality.assess(video(None), [seg(0, 0, 5, 3)])
    assert q.words_per_minute is None and q.coverage is None
    assert q.flags == []


def test_no_segments_is_flagged():
    q = quality.assess(video(), [])
    assert q.flags == ["no transcript text"]
    assert q.words == 0


def test_to_dict_is_json_friendly():
    import json

    q = quality.assess(video(), [seg(0, 0, 10, 100), seg(1, 40, 60, 100)])
    data = json.loads(json.dumps(q.to_dict()))
    assert data["gaps"] == [{"start": 10, "end": 40, "seconds": 30.0}]
    assert data["words"] == 200
    assert data["flags"]


def test_format_time():
    assert quality.format_time(5) == "0:05"
    assert quality.format_time(125) == "2:05"
    assert quality.format_time(3725) == "1:02:05"
    assert quality.format_time(-4) == "0:00"
