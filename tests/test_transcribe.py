from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bytx import transcribe
from bytx.config import AsrConfig, AsrModelConfig

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

    def __init__(self, model_dir, device, compute_type):
        if (device, compute_type) in _FakeWhisperModel.fail_on_construct:
            raise RuntimeError(f"cannot construct on {device}/{compute_type}")
        self.device = device
        self.compute_type = compute_type

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

        return iter([Seg(0.0, 1.0, "hello"), Seg(1.0, 2.0, "না না না না না")]), info


@pytest.fixture(autouse=True)
def _patch_whisper_model(monkeypatch, tmp_path):
    _FakeWhisperModel.fail_on_construct = set()
    _FakeWhisperModel.fail_on_transcribe = set()
    monkeypatch.setattr(transcribe, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcribe, "ensure_cuda_dll_paths", lambda: [])
    (tmp_path / "model_marker").write_text("x")
    yield


def make_transcriber(tmp_path, device="auto", compute_type="float16"):
    model_cfg = AsrModelConfig(key="m", source="org/m", language="bn")
    asr_cfg = AsrConfig(
        model="m", device=device, compute_type=compute_type, beam_size=1,
        vad_filter=False, vad_min_silence_ms=500, without_timestamps=True, condition_on_previous_text=False,
        repetition_penalty=1.1, no_repeat_ngram_size=3, initial_prompt="", models={"m": model_cfg},
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


def test_transcribe_collapses_repeats_and_assigns_sequential_idx(tmp_path):
    t = make_transcriber(tmp_path)
    result = t.transcribe(Path("fake_audio.wav"))
    assert result.device == "cuda"
    assert [s.idx for s in result.segments] == [0, 1]
    assert result.segments[1].text == "না না না"  # collapsed from 5 repeats to max_repeats=3
    assert result.language == "bn"


def test_unload_clears_model_and_reloads_on_next_use(tmp_path):
    t = make_transcriber(tmp_path)
    _ = t.device
    assert t._model is not None
    t.unload()
    assert t._model is None
    # Using it again should transparently reload.
    assert t.device == "cuda"
