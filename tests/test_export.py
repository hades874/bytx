from __future__ import annotations

import json

import pytest

from bytx import export, store


def make_video(**overrides) -> store.Video:
    defaults = dict(
        id="abc123", url="https://youtu.be/abc123", title="বাংলা লেকচার",
        channel="Test Channel", channel_id="c1", upload_date="20260101",
        duration=125.0, source="asr", asr_model="tugstugi-medium-bn",
    )
    defaults.update(overrides)
    return store.Video(**defaults)


def make_segments() -> list[store.Segment]:
    return [
        store.Segment(idx=0, start_s=0.0, end_s=2.5, text="প্রথম লাইন"),
        store.Segment(idx=1, start_s=2.5, end_s=65.25, text="দ্বিতীয় লাইন"),
    ]


# --- timestamp formatting -----------------------------------------------------

def test_srt_timestamp_format():
    assert export._srt_timestamp(0) == "00:00:00,000"
    assert export._srt_timestamp(65.25) == "00:01:05,250"
    assert export._srt_timestamp(3661.001) == "01:01:01,001"


def test_srt_timestamp_negative_clamped_to_zero():
    assert export._srt_timestamp(-1.0) == "00:00:00,000"


def test_vtt_timestamp_format():
    assert export._vtt_timestamp(0) == "00:00:00.000"
    assert export._vtt_timestamp(65.25) == "00:01:05.250"


# --- to_txt / to_srt / to_vtt --------------------------------------------------

def test_to_txt_one_line_per_segment():
    text = export.to_txt(make_segments())
    assert text == "প্রথম লাইন\nদ্বিতীয় লাইন\n"


def test_to_txt_empty():
    assert export.to_txt([]) == ""


def test_to_srt_structure():
    srt = export.to_srt(make_segments())
    lines = srt.split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "প্রথম লাইন"
    assert "2" in lines
    assert "00:00:02,500 --> 00:01:05,250" in srt


def test_to_vtt_structure():
    vtt = export.to_vtt(make_segments())
    lines = vtt.split("\n")
    assert lines[0] == "WEBVTT"
    assert "00:00:00.000 --> 00:00:02.500" in vtt
    assert "প্রথম লাইন" in vtt


def test_to_json_roundtrip():
    video = make_video()
    segs = make_segments()
    raw = export.to_json(video, segs, summary="সারাংশ")
    data = json.loads(raw)
    assert data["id"] == "abc123"
    assert data["title"] == "বাংলা লেকচার"
    assert data["summary"] == "সারাংশ"
    assert len(data["segments"]) == 2
    assert data["segments"][0] == {"start": 0.0, "end": 2.5, "text": "প্রথম লাইন"}


def test_render_dispatch_and_unknown_format():
    video = make_video()
    segs = make_segments()
    assert export.render("txt", video, segs) == export.to_txt(segs)
    assert export.render("srt", video, segs) == export.to_srt(segs)
    assert export.render("vtt", video, segs) == export.to_vtt(segs)
    assert json.loads(export.render("json", video, segs))["id"] == "abc123"
    with pytest.raises(ValueError):
        export.render("docx", video, segs)


# --- filesystem writers ---------------------------------------------------

def test_write_all_creates_expected_files(tmp_path):
    video = make_video()
    segs = make_segments()
    written = export.write_all(tmp_path, video, segs, summary="সারাংশ")

    assert set(written) == {"txt", "srt", "vtt", "json"}
    out_dir = tmp_path / "abc123"
    assert (out_dir / "transcript.txt").exists()
    assert (out_dir / "transcript.srt").exists()
    assert (out_dir / "transcript.vtt").exists()
    assert (out_dir / "transcript.json").exists()

    txt_content = (out_dir / "transcript.txt").read_text(encoding="utf-8")
    assert "প্রথম লাইন" in txt_content

    json_content = json.loads((out_dir / "transcript.json").read_text(encoding="utf-8"))
    assert json_content["summary"] == "সারাংশ"


def test_write_info_json(tmp_path):
    video = make_video()
    path = export.write_info_json(tmp_path, video)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"] == "abc123"
    assert data["channel"] == "Test Channel"
    assert "segments" not in data


def test_write_summary_md(tmp_path):
    video = make_video()
    path = export.write_summary_md(tmp_path, video, "এই ভিডিওর সারাংশ")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# বাংলা লেকচার")
    assert "এই ভিডিওর সারাংশ" in content
    assert video.url in content


def test_write_summary_md_falls_back_to_id_when_no_title(tmp_path):
    video = make_video(title=None)
    path = export.write_summary_md(tmp_path, video, "সারাংশ")
    content = path.read_text(encoding="utf-8")
    assert content.startswith("# abc123")


# --- sanitize_folder_name ---------------------------------------------------

def test_sanitize_folder_name_basic():
    assert export.sanitize_folder_name("Some Channel", "Some Title", "abc123") == "Some Channel - Some Title"


def test_sanitize_folder_name_replaces_illegal_chars():
    name = export.sanitize_folder_name('Chan<>:"/\\|?*Name', "Ti|tle?", "abc123")
    for ch in '<>:"/\\|?*':
        assert ch not in name


def test_sanitize_folder_name_strips_trailing_dots_and_spaces():
    name = export.sanitize_folder_name("Channel. ", " Title...", "abc123")
    assert not name.endswith(".")
    assert not name.endswith(" ")


def test_sanitize_folder_name_truncates_long_combined_name():
    name = export.sanitize_folder_name("C" * 100, "T" * 200, "abc123")
    assert len(name) <= 150
    assert not name.endswith(".")
    assert not name.endswith(" ")


def test_sanitize_folder_name_falls_back_to_video_id_when_empty():
    assert export.sanitize_folder_name("", "", "abc123") == "abc123 - abc123"
    assert export.sanitize_folder_name(None, None, "abc123") == "abc123 - abc123"
    assert export.sanitize_folder_name("...", "***", "abc123") == "abc123 - abc123"


# --- write_transcript_copy ---------------------------------------------------

def test_write_transcript_copy_writes_plain_text_only(tmp_path):
    video = make_video()
    segs = make_segments()
    out_dir = export.write_transcript_copy(tmp_path, video, segs)

    assert out_dir == tmp_path / "Test Channel - বাংলা লেকচার"
    assert (out_dir / "transcript.txt").read_text(encoding="utf-8") == export.to_txt(segs)
    assert (out_dir / ".video_id").read_text(encoding="utf-8") == "abc123"
    assert not (out_dir / "transcript.srt").exists()
    assert not (out_dir / "transcript.json").exists()


def test_write_transcript_copy_rerun_same_video_overwrites_in_place(tmp_path):
    video = make_video()
    export.write_transcript_copy(tmp_path, video, make_segments())
    out_dir = export.write_transcript_copy(tmp_path, video, [store.Segment(idx=0, start_s=0, end_s=1, text="নতুন")])

    assert out_dir == tmp_path / "Test Channel - বাংলা লেকচার"
    assert set(p.name for p in tmp_path.iterdir()) == {"Test Channel - বাংলা লেকচার"}
    assert (out_dir / "transcript.txt").read_text(encoding="utf-8") == "নতুন\n"


def test_write_transcript_copy_collision_appends_video_id(tmp_path):
    video_a = make_video(id="abc123")
    video_b = make_video(id="xyz789")
    dir_a = export.write_transcript_copy(tmp_path, video_a, make_segments())
    dir_b = export.write_transcript_copy(tmp_path, video_b, make_segments())

    assert dir_a != dir_b
    assert dir_a == tmp_path / "Test Channel - বাংলা লেকচার"
    assert dir_b == tmp_path / "Test Channel - বাংলা লেকচার (xyz789)"


def test_write_transcript_copy_title_change_reuses_original_folder(tmp_path):
    original = export.write_transcript_copy(tmp_path, make_video(), make_segments())
    renamed = export.write_transcript_copy(tmp_path, make_video(title="নতুন শিরোনাম"), make_segments())

    assert renamed == original
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_write_transcript_copy_ignores_folders_without_marker(tmp_path):
    (tmp_path / "unrelated").mkdir()
    out_dir = export.write_transcript_copy(tmp_path, make_video(), make_segments())

    assert out_dir == tmp_path / "Test Channel - বাংলা লেকচার"
