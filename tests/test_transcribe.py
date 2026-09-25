from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bytx import transcribe
from bytx.config import AsrConfig, AsrModelConfig, load_config

SR = transcribe.SAMPLE_RATE

# --- collapse_repeated_ngrams -------------------------------------------------

def test_collapse_no_repeats_unchanged():
    text = "আমি আজকে বাজারে গিয়েছিলাম"
    assert transcribe.collapse_repeated_ngrams(text) == text


def test_collapse_single_word_repeats():
    text = "না " * 10
    result = transcribe.collapse_repeated_ngrams(text.strip(), max_repeats=3)
    assert result == "না না না"


def test_collapse_phrase_repeats():
    text = ("ঠিক আছে " * 8).strip()
    result = transcribe.collapse_repeated_ngrams(text, max_repeats=3)
    assert result == "ঠিক আছে ঠিক আছে ঠিক আছে"


def test_collapse_mixed_text_keeps_normal_part():
    text = "শুরুতে ভালো কথা " + ("হ্যাঁ " * 9).strip()
    result = transcribe.collapse_repeated_ngrams(text, max_repeats=3)
    assert result.startswith("শুরুতে ভালো কথা")
    assert result.count("হ্যাঁ") == 3


def test_collapse_short_text_untouched():
    assert transcribe.collapse_repeated_ngrams("এক দুই") == "এক দুই"


def test_collapse_empty_string():
    assert transcribe.collapse_repeated_ngrams("") == ""


# --- Transcriber device/compute_type fallback (WhisperModel mocked) ---------

class _FakeWhisperModel:
    """Fails to construct or to transcribe for specific (device, compute_type)
    combos, to exercise the fallback chain deterministically offline.
    """

    fail_on_construct: set[tuple[str, str]] = set()
    fail_on_transcribe: set[tuple[str, str]] = set()
    # Maps a decoded clip (np array) to its text; replaced per test.
    respond = staticmethod(lambda clip: "hello")
    clip_lengths: list[int] = []

    class _Tokenizer:
        """One token per whitespace-separated word."""

        def encode(self, text, add_special_tokens=True):
            class Enc:
                pass

            enc = Enc()
            enc.ids = list(range(len(text.split())))
            return enc

    def __init__(self, model_dir, device, compute_type):
        if (device, compute_type) in _FakeWhisperModel.fail_on_construct:
            raise RuntimeError(f"cannot construct on {device}/{compute_type}")
        self.device = device
        self.compute_type = compute_type
        self.hf_tokenizer = _FakeWhisperModel._Tokenizer()

    def transcribe(self, audio, language=None, beam_size=1, **kwargs):
        if (self.device, self.compute_type) in _FakeWhisperModel.fail_on_transcribe:
            raise RuntimeError(f"garbage output on {self.device}/{self.compute_type}")

        class Info:
            pass

        info = Info()
        info.language = language or "bn"

        if isinstance(audio, np.ndarray) and audio.shape[0] <= 16000:
            return iter([]), info  # smoke test: no speech in silence

        class Seg:
            def __init__(self, start, end, text):
                self.start, self.end, self.text = start, end, text

        _FakeWhisperModel.clip_lengths.append(len(audio))
        return iter([Seg(0.0, len(audio) / SR, _FakeWhisperModel.respond(audio))]), info


@pytest.fixture(autouse=True)
def _patch_whisper_model(monkeypatch, tmp_path):
    _FakeWhisperModel.fail_on_construct = set()
    _FakeWhisperModel.fail_on_transcribe = set()
    _FakeWhisperModel.respond = staticmethod(lambda clip: "hello")
    _FakeWhisperModel.clip_lengths = []
    monkeypatch.setattr(transcribe, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcribe, "ensure_cuda_dll_paths", lambda: [])
    (tmp_path / "model_marker").write_text("x")
    yield


def make_transcriber(tmp_path, device="auto", compute_type="float16", vad_filter=False,
                     max_clip_seconds=12.0):
    model_cfg = AsrModelConfig(key="m", source="org/m", language="bn")
    asr_cfg = AsrConfig(
        model="m", device=device, compute_type=compute_type, beam_size=1,
        vad_filter=vad_filter, vad_min_silence_ms=500, without_timestamps=True,
        condition_on_previous_text=False, repetition_penalty=1.1, no_repeat_ngram_size=3,
        initial_prompt="", models={"m": model_cfg}, max_clip_seconds=max_clip_seconds,
    )
    return transcribe.Transcriber(tmp_path, model_cfg, asr_cfg)


def test_loads_with_requested_compute_type_when_it_works(tmp_path):
    t = make_transcriber(tmp_path)
    assert t.device == "cuda"
    assert t.compute_type == "float16"


def test_falls_back_to_int8_float16_when_float16_construction_fails(tmp_path):
    _FakeWhisperModel.fail_on_construct = {("cuda", "float16")}
    t = make_transcriber(tmp_path)
    assert t.device == "cuda"
    assert t.compute_type == "int8_float16"


def test_falls_back_when_float16_construct_ok_but_smoke_test_transcribe_fails(tmp_path):
    _FakeWhisperModel.fail_on_transcribe = {("cuda", "float16")}
    t = make_transcriber(tmp_path)
    assert t.compute_type == "int8_float16"


def test_falls_back_to_cpu_when_all_cuda_options_fail(tmp_path):
    _FakeWhisperModel.fail_on_construct = {
        ("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"),
    }
    t = make_transcriber(tmp_path)
    assert t.device == "cpu"
    assert t.compute_type in ("int8", "float32")


def test_raises_model_load_error_when_nothing_works(tmp_path):
    _FakeWhisperModel.fail_on_construct = {
        ("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "int8"),
        ("cpu", "int8"), ("cpu", "float32"),
    }
    t = make_transcriber(tmp_path)
    with pytest.raises(transcribe.ModelLoadError):
        _ = t.device


def test_explicit_device_does_not_try_other_devices(tmp_path):
    _FakeWhisperModel.fail_on_construct = {("cpu", "int8"), ("cpu", "float32")}
    t = make_transcriber(tmp_path, device="cpu", compute_type="int8")
    # cpu/int8 fails to construct, but since device is pinned to cpu only,
    # it should still try int8 -> float32 fallback within cpu, not cuda.
    with pytest.raises(transcribe.ModelLoadError) as exc_info:
        _ = t.device
    assert "cuda" not in str(exc_info.value)


def test_missing_model_dir_raises_before_touching_gpu(tmp_path):
    model_cfg = AsrModelConfig(key="m", source="org/m", language="bn")
    asr_cfg = AsrConfig(
        model="m", device="auto", compute_type="float16", beam_size=1,
        vad_filter=False, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
        repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="", models={"m": model_cfg},
    )
    t = transcribe.Transcriber(tmp_path / "does_not_exist", model_cfg, asr_cfg)
    with pytest.raises(transcribe.ModelLoadError, match="does not exist"):
        _ = t.device


def _silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _patch_audio(monkeypatch, audio, vad_chunks=None):
    monkeypatch.setattr(transcribe, "decode_audio", lambda path, sampling_rate=SR: audio)
    if vad_chunks is not None:
        monkeypatch.setattr(transcribe, "get_speech_timestamps", lambda a, opts: vad_chunks)


def test_transcribe_collapses_repeats_and_assigns_sequential_idx(tmp_path, monkeypatch):
    # 20 s with no VAD -> one over-long range, force-split into two clips.
    _patch_audio(monkeypatch, _silence(20))
    texts = iter(["hello", "না না না না না"])
    _FakeWhisperModel.respond = staticmethod(lambda clip: next(texts))
    t = make_transcriber(tmp_path)
    result = t.transcribe(Path("fake_audio.wav"))
    assert result.device == "cuda"
    assert [s.idx for s in result.segments] == [0, 1]
    assert result.segments[1].text == "না না না"  # collapsed from 5 repeats to max_repeats=3
    assert result.language == "bn"


def test_transcribe_uses_real_clip_times_and_skips_silence(tmp_path, monkeypatch):
    chunks = [
        {"start": 2 * SR, "end": 5 * SR},
        {"start": 20 * SR, "end": 24 * SR},
    ]
    _patch_audio(monkeypatch, _silence(30), vad_chunks=chunks)
    t = make_transcriber(tmp_path, vad_filter=True)
    result = t.transcribe(Path("fake_audio.wav"))
    assert [(s.start_s, s.end_s) for s in result.segments] == [(2.0, 5.0), (20.0, 24.0)]
    assert result.stats.clips == 2
    assert result.stats.speech_seconds == 7.0
    assert result.stats.audio_seconds == 30.0
    assert _FakeWhisperModel.clip_lengths == [3 * SR, 4 * SR]  # never the whole file


def test_transcribe_no_speech_gives_no_segments(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10), vad_chunks=[])
    result = make_transcriber(tmp_path, vad_filter=True).transcribe(Path("x.wav"))
    assert result.segments == []
    assert result.stats.clips == 0


def _long_when_over(seconds, words=220):
    return staticmethod(
        lambda clip: " ".join(f"শব্দ{i}" for i in range(words))
        if len(clip) > seconds * SR
        else "ছোট লেখা"
    )


def test_transcribe_resplits_clip_that_hits_token_limit(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10))
    _FakeWhisperModel.respond = _long_when_over(6)  # a 10 s clip looks truncated, 5 s halves don't
    result = make_transcriber(tmp_path).transcribe(Path("x.wav"))
    assert len(result.segments) == 2
    first, second = result.segments
    assert first.start_s == 0.0
    assert first.end_s == second.start_s  # contiguous: nothing dropped between halves
    assert second.end_s == 10.0
    assert result.stats.resplit == 1
    assert result.stats.truncated_unresolved == 0


def test_transcribe_counts_clips_still_truncated_at_minimum_size(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10))
    _FakeWhisperModel.respond = _long_when_over(0)  # every clip looks truncated
    result = make_transcriber(tmp_path).transcribe(Path("x.wav"))
    stats = result.stats
    assert stats.resplit >= 2
    assert stats.truncated_unresolved == len(result.segments) >= 2  # every leaf is a floor-size clip
    assert result.segments[0].start_s == 0.0 and result.segments[-1].end_s == 10.0
    for a, b in zip(result.segments, result.segments[1:], strict=False):
        assert a.end_s == b.start_s  # splitting never drops audio between pieces
    assert all(s.end_s - s.start_s < 2 * transcribe.MIN_CLIP_S for s in result.segments)


def test_plan_clips_merges_up_to_limit_and_keeps_gaps():
    chunks = [
        {"start": 0, "end": 3 * SR},
        {"start": 4 * SR, "end": 8 * SR},
        {"start": 9 * SR, "end": 13 * SR},  # would make the first clip 13 s > 12
        {"start": 14 * SR, "end": 15 * SR},
    ]
    assert transcribe.plan_clips(chunks, 12) == [(0, 8 * SR), (9 * SR, 15 * SR)]


def test_plan_clips_empty_and_oversized_single_chunk():
    assert transcribe.plan_clips([], 12) == []
    assert transcribe.plan_clips([{"start": 0, "end": 20 * SR}], 12) == [(0, 20 * SR)]


def test_find_split_point_picks_the_quiet_gap():
    n = 10 * SR
    audio = np.full(n, 0.5, dtype=np.float32)
    audio[int(0.55 * n) : int(0.55 * n) + SR // 5] = 0.0  # 200 ms pause at 55%
    idx = transcribe.find_split_point(audio)
    assert int(0.55 * n) <= idx <= int(0.55 * n) + SR // 5


def test_find_split_point_stays_inside_the_clip_when_uniform():
    audio = np.full(4 * SR, 0.3, dtype=np.float32)
    assert 0 < transcribe.find_split_point(audio) < len(audio)


# --- config ---------------------------------------------------------------

def test_shipped_config_has_clip_settings_and_fast_decoding_defaults():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.toml")
    assert cfg.asr.max_clip_seconds == 12
    assert cfg.asr.vad_threshold == 0.5
    assert cfg.asr.vad_speech_pad_ms == 400
    # Keep the fallbacks (used when a key is missing) equal to the shipped values.
    assert (cfg.asr.beam_size, cfg.asr.repetition_penalty, cfg.asr.no_repeat_ngram_size) == (1, 1.0, 0)


def test_config_defaults_when_asr_keys_missing(tmp_path):
    (tmp_path / "config.toml").write_text("[asr]\nmodel = 'm'\n", encoding="utf-8")
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.asr.max_clip_seconds == 12.0
    assert cfg.asr.beam_size == 1
    assert cfg.asr.repetition_penalty == 1.0
    assert cfg.asr.no_repeat_ngram_size == 0


def test_unload_clears_model_and_reloads_on_next_use(tmp_path):
    t = make_transcriber(tmp_path)
    _ = t.device
    assert t._model is not None
    t.unload()
    assert t._model is None
    # Using it again should transparently reload.
    assert t.device == "cuda"
