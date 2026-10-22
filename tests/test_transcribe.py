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
    # Language passed to each real (non smoke-test) decode.
    languages: list[str | None] = []
    # initial_prompt passed to each real decode.
    prompts: list[str | None] = []
    # (avg_logprob, no_speech_prob) reported for each decoded clip.
    confidence = (-0.3, 0.02)

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
                self.avg_logprob, self.no_speech_prob = _FakeWhisperModel.confidence

        _FakeWhisperModel.clip_lengths.append(len(audio))
        _FakeWhisperModel.languages.append(language)
        _FakeWhisperModel.prompts.append(kwargs.get("initial_prompt"))
        return iter([Seg(0.0, len(audio) / SR, _FakeWhisperModel.respond(audio))]), info



@pytest.fixture(autouse=True)
def _patch_whisper_model(monkeypatch, tmp_path):
    _FakeWhisperModel.fail_on_construct = set()
    _FakeWhisperModel.fail_on_transcribe = set()
    _FakeWhisperModel.respond = staticmethod(lambda clip: "hello")
    _FakeWhisperModel.clip_lengths = []
    _FakeWhisperModel.languages = []
    _FakeWhisperModel.prompts = []
    _FakeWhisperModel.confidence = (-0.3, 0.02)
    monkeypatch.setattr(transcribe, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(transcribe, "ensure_cuda_dll_paths", lambda: [])
    (tmp_path / "model_marker").write_text("x")
    yield


def make_transcriber(tmp_path, device="auto", compute_type="float16", vad_filter=False,
                     max_clip_seconds=12.0, language="bn", model_max_clip_seconds=None,
                     glossary=None, initial_prompt="", prompt_languages=("en",)):
    """`language` pins the model to a language; "" makes it multilingual."""
    model_cfg = AsrModelConfig(
        key="m", source="org/m", language=language, max_clip_seconds=model_max_clip_seconds
    )
    asr_cfg = AsrConfig(
        model="m", device=device, compute_type=compute_type, beam_size=1,
        vad_filter=vad_filter, vad_min_silence_ms=500, without_timestamps=True,
        condition_on_previous_text=False, repetition_penalty=1.1, no_repeat_ngram_size=3,
        initial_prompt=initial_prompt, models={"m": model_cfg}, max_clip_seconds=max_clip_seconds,
        glossary_prompt_languages=prompt_languages,
    )
    return transcribe.Transcriber(tmp_path, model_cfg, asr_cfg, glossary)


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


def test_transcribe_stores_the_clips_confidence_on_each_segment(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    _FakeWhisperModel.confidence = (-0.75, 0.31)
    result = make_transcriber(tmp_path).transcribe(Path("x.wav"))
    assert [(s.avg_logprob, s.no_speech_prob) for s in result.segments] == [(-0.75, 0.31)]


def test_weighted_logprob_weights_by_segment_length_and_handles_no_segments():
    class Seg:
        def __init__(self, start, end, lp):
            self.start, self.end, self.avg_logprob = start, end, lp

    assert transcribe._weighted_logprob([]) is None
    assert transcribe._weighted_logprob([Seg(0, 1, -1.0), Seg(1, 4, -0.2)]) == pytest.approx(-0.4)
    assert transcribe._weighted_logprob([Seg(0, 0, -1.0), Seg(0, 0, -0.5)]) == pytest.approx(-0.75)


def _glossary():
    from bytx import glossary

    return glossary.parse(["TrustEd Ireland <= Trusted Ireland", "IRCC"])


def test_glossary_terms_prime_the_decoder_for_enabled_languages(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    t = make_transcriber(
        tmp_path, language="", glossary=_glossary(), initial_prompt="visa talk"
    )
    t.transcribe(Path("x.wav"), "en")
    assert _FakeWhisperModel.prompts == ["visa talk, TrustEd Ireland, IRCC"]


def test_glossary_does_not_prime_languages_that_did_not_opt_in(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    make_transcriber(tmp_path, glossary=_glossary()).transcribe(Path("x.wav"))  # pinned to bn
    assert _FakeWhisperModel.prompts == [None]


def test_no_prompt_without_glossary_or_initial_prompt(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    make_transcriber(tmp_path, language="").transcribe(Path("x.wav"), "en")
    assert _FakeWhisperModel.prompts == [None]


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

def test_shipped_config_has_clip_settings_and_fast_decoding_defaults(tmp_path):
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    cfg = load_config(tmp_path / "config.toml")
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


# --- languages ----------------------------------------------------------------

def test_multilingual_model_decodes_in_the_requested_language(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    t = make_transcriber(tmp_path, language="")
    result = t.transcribe(Path("x.wav"), "en")
    assert result.language == "en"
    assert _FakeWhisperModel.languages == ["en"]


def test_pinned_model_uses_its_own_language_when_none_is_passed(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    result = make_transcriber(tmp_path, language="bn").transcribe(Path("x.wav"))
    assert result.language == "bn"
    assert _FakeWhisperModel.languages == ["bn"]


def test_pinned_model_refuses_a_different_language(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    with pytest.raises(ValueError, match="pinned to 'bn' and cannot transcribe 'en'"):
        make_transcriber(tmp_path, language="bn").transcribe(Path("x.wav"), "en")
    assert _FakeWhisperModel.languages == []  # refused before decoding anything


def test_multilingual_model_needs_a_language(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(5))
    with pytest.raises(ValueError, match="multilingual"):
        make_transcriber(tmp_path, language="").transcribe(Path("x.wav"))


def test_smoke_test_works_for_multilingual_models(tmp_path):
    # A multilingual model has no pinned language to smoke-test with; loading must still work.
    assert make_transcriber(tmp_path, language="").device == "cuda"


def test_model_max_clip_seconds_overrides_the_global_setting(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10))  # no VAD: one 10 s range
    # Global limit 12 s keeps the range whole ...
    assert len(make_transcriber(tmp_path, max_clip_seconds=12.0).transcribe(Path("x.wav")).segments) == 1
    # ... but a model limit of 6 s (x1.25 slack = 7.5 s) forces a split.
    t = make_transcriber(tmp_path, max_clip_seconds=12.0, model_max_clip_seconds=6.0)
    assert t.max_clip_seconds == 6.0
    assert len(t.transcribe(Path("x.wav")).segments) == 2


def test_token_limit_fallback_depends_on_language(tmp_path):
    t = make_transcriber(tmp_path)
    t._model = object()  # no hf_tokenizer -> characters-per-token estimate
    text = "a" * 400
    assert t._nears_token_limit(text, "bn")  # 400 * 0.75 = 300 tokens
    assert not t._nears_token_limit(text, "en")  # 400 * 0.3 = 120 tokens
    assert not t._nears_token_limit("a" * 100, "xx")  # unknown language: 0.5 -> 50 tokens


# --- TranscriberPool -------------------------------------------------------------

def _pool_config(tmp_path, extra=""):
    for key in ("bn-model", "multi"):  # local CTranslate2 folders, used as they are
        (tmp_path / "models" / key).mkdir(parents=True)
        (tmp_path / "models" / key / "model.bin").write_bytes(b"")
    models = (tmp_path / "models").as_posix()
    (tmp_path / "config.toml").write_text(
        f"""
[asr]
model = "bn-model"
[asr.language_models]
bn = "bn-model"
en = "multi"
[asr.models.bn-model]
source = "{models}/bn-model"
language = "bn"
[asr.models.multi]
source = "{models}/multi"
language = ""
"""
        + extra,
        encoding="utf-8",
    )
    return load_config(tmp_path / "config.toml")


def test_pool_routes_languages_to_their_models(tmp_path):
    pool = transcribe.TranscriberPool(_pool_config(tmp_path))
    assert pool.for_language("bn").model_cfg.key == "bn-model"
    assert pool.for_language("en").model_cfg.key == "multi"


def test_pool_reuses_the_loaded_model_for_the_same_key(tmp_path):
    pool = transcribe.TranscriberPool(_pool_config(tmp_path))
    english = pool.for_language("en")
    assert pool.for_language("en") is english  # same model: no reload


def test_pool_unloads_the_previous_model_before_switching(tmp_path):
    cfg = _pool_config(tmp_path)
    pool = transcribe.TranscriberPool(cfg)
    bangla = pool.for_language("bn")
    _ = bangla.device  # load it
    assert bangla._model is not None

    pool.for_language("en")

    assert bangla._model is None  # freed before the other model is used


def test_pool_unload_drops_the_current_model(tmp_path):
    cfg = _pool_config(tmp_path)
    pool = transcribe.TranscriberPool(cfg)
    model = pool.for_language("en")
    _ = model.device
    pool.unload()
    assert model._model is None
    assert pool.for_language("en") is not model  # a fresh Transcriber is built on next use


def test_pool_unmapped_language_raises_clear_error(tmp_path):
    pool = transcribe.TranscriberPool(_pool_config(tmp_path))
    with pytest.raises(ValueError, match="No ASR model is configured for language 'fr'"):
        pool.for_language("fr")


# --- sample_text (used by language detection) -----------------------------------------

def test_evenly_spaced_returns_everything_when_there_are_few_items():
    assert transcribe.evenly_spaced([1, 2, 3], 5) == [1, 2, 3]
    assert transcribe.evenly_spaced([], 5) == []


def test_evenly_spaced_includes_first_and_last_and_spreads_the_rest():
    assert transcribe.evenly_spaced(list(range(100)), 5) == [0, 25, 50, 74, 99]
    picks = transcribe.evenly_spaced(list(range(10)), 3)
    assert picks[0] == 0 and picks[-1] == 9 and len(picks) == 3


def test_evenly_spaced_edge_sizes():
    assert transcribe.evenly_spaced(list(range(9)), 1) == [4]
    assert transcribe.evenly_spaced(list(range(9)), 0) == []
    assert transcribe.evenly_spaced(list(range(4)), 3) == [0, 2, 3]  # picks never repeat


def _spread_chunks(n, gap_s=30, len_s=5):
    return [{"start": i * gap_s * SR, "end": (i * gap_s + len_s) * SR} for i in range(n)]


def test_sample_text_decodes_only_a_spread_of_clips(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(40 * 30), vad_chunks=_spread_chunks(40))
    t = make_transcriber(tmp_path, vad_filter=True)
    text = t.sample_text(Path("x.wav"), clips=6)
    assert len(_FakeWhisperModel.clip_lengths) == 6  # not all 40
    assert text == "hello " * 5 + "hello"


def test_sample_text_covers_the_start_and_the_end_of_the_video(tmp_path, monkeypatch):
    audio = np.zeros(40 * 30 * SR, dtype=np.float32)
    for c in _spread_chunks(40):  # tag each speech chunk with its position
        audio[c["start"] : c["end"]] = 0.5 + c["start"] / (40 * 30 * SR) / 10
    _patch_audio(monkeypatch, audio, vad_chunks=_spread_chunks(40))
    seen = []
    _FakeWhisperModel.respond = staticmethod(lambda clip: seen.append(float(clip[0])) or "x")
    make_transcriber(tmp_path, vad_filter=True).sample_text(Path("x.wav"), clips=5)
    assert seen[0] == pytest.approx(0.5)  # first chunk
    assert seen[-1] > 0.59  # last chunk (0.5 + 1170/1200/10)


def test_sample_text_uses_the_pinned_language(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10), vad_chunks=_spread_chunks(1))
    make_transcriber(tmp_path, vad_filter=True, language="bn").sample_text(Path("x.wav"))
    assert _FakeWhisperModel.languages == ["bn"]


def test_sample_text_of_a_silent_video_is_empty(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10), vad_chunks=[])
    assert make_transcriber(tmp_path, vad_filter=True).sample_text(Path("x.wav")) == ""
    assert _FakeWhisperModel.clip_lengths == []


def test_sample_text_refuses_a_language_the_model_is_not_pinned_to(tmp_path, monkeypatch):
    _patch_audio(monkeypatch, _silence(10), vad_chunks=_spread_chunks(1))
    with pytest.raises(ValueError, match="pinned to 'bn'"):
        make_transcriber(tmp_path, vad_filter=True, language="bn").sample_text(Path("x.wav"), "en")
