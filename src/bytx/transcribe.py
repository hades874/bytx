"""faster-whisper wrapper: loads a converted CTranslate2 Whisper model,
transcribes audio with VAD, and applies anti-hallucination settings.

Handles a known quirk on some GTX 16xx cards where fp16 CUDA kernels
silently produce empty/garbage output: on load we run a 1-second silence
smoke test and fall back down a compute_type chain (and finally to CPU)
if it fails, rather than failing the whole run.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from . import store
from .config import AsrConfig, AsrModelConfig
from .cuda_dll import ensure_cuda_dll_paths

logger = logging.getLogger(__name__)

# Compute types to try, in order, per device, after the configured one fails.
_FALLBACK_CHAIN: dict[str, list[str]] = {
    "cuda": ["float16", "int8_float16", "int8"],
    "cpu": ["int8", "float32"],
}


def collapse_repeated_ngrams(text: str, max_repeats: int = 3) -> str:
    """Collapses runs of the same word/short phrase repeated many times in a
    row -- a common Whisper hallucination on silence, music, or noise (e.g.
    "না না না না না না না না..."). Keeps at most `max_repeats` copies.
    """
    words = text.split()
    n = len(words)
    if n <= max_repeats:
        return text  # can't have more than max_repeats occurrences of anything

    out: list[str] = []
    i = 0
    while i < n:
        matched = False
        for phrase_len in range(1, 5):
            if i + phrase_len * (max_repeats + 1) > n:
                continue
            phrase = words[i : i + phrase_len]
            repeats = 1
            j = i + phrase_len
            while words[j : j + phrase_len] == phrase:
                repeats += 1
                j += phrase_len
            if repeats > max_repeats:
                out.extend(phrase * max_repeats)
                i = j
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1
    return " ".join(out)


@dataclass
class TranscribeResult:
    segments: list[store.Segment]
    device: str
    compute_type: str
    language: str | None


class ModelLoadError(RuntimeError):
    pass


class Transcriber:
    """Lazily loads the model on first use; call `unload()` between the ASR
    stage and later GPU-using stages (e.g. Ollama) to free VRAM.
    """

    def __init__(self, model_dir: Path, model_cfg: AsrModelConfig, asr_cfg: AsrConfig):
        self.model_dir = model_dir
        self.model_cfg = model_cfg
        self.asr_cfg = asr_cfg
        self._model: WhisperModel | None = None
        self._device: str | None = None
        self._compute_type: str | None = None

    def _candidate_devices(self) -> list[str]:
        if self.asr_cfg.device == "auto":
            return ["cuda", "cpu"]
        return [self.asr_cfg.device]

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.exists():
            raise ModelLoadError(
                f"Model directory {self.model_dir} does not exist. "
                f"Run `bytx setup` (or scripts/convert_model.py {self.model_cfg.key}) first."
            )
        ensure_cuda_dll_paths()

        errors: list[str] = []
        requested = self.asr_cfg.compute_type
        for device in self._candidate_devices():
            # The configured compute_type (e.g. float16) is a GPU-oriented
            # choice; only try it first on cuda. On cpu (or any other
            # fallback device), use that device's own default chain instead
            # of dragging a GPU-tuned setting along.
            if device == "cuda":
                chain = [requested] + [ct for ct in _FALLBACK_CHAIN.get(device, []) if ct != requested]
            else:
                chain = list(_FALLBACK_CHAIN.get(device, [requested]))
            for compute_type in chain:
                try:
                    model = WhisperModel(str(self.model_dir), device=device, compute_type=compute_type)
                    self._smoke_test(model)
                except Exception as exc:  # noqa: BLE001 - intentionally broad; we try the next option
                    errors.append(f"{device}/{compute_type}: {exc}")
                    continue
                self._model = model
                self._device = device
                self._compute_type = compute_type
                if (device, compute_type) != ("cuda", requested):
                    logger.warning(
                        "ASR using device=%s compute_type=%s (requested cuda/%s; see errors for why)",
                        device, compute_type, requested,
                    )
                return
        raise ModelLoadError(
            "Could not load ASR model on any device/compute_type combination:\n" + "\n".join(errors)
        )

    def _smoke_test(self, model: WhisperModel) -> None:
        silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz
        segments, _info = model.transcribe(silence, language=self.model_cfg.language, beam_size=1)
        list(segments)  # force generator to run

    @property
    def device(self) -> str:
        self._load()
        assert self._device is not None
        return self._device

    @property
    def compute_type(self) -> str:
        self._load()
        assert self._compute_type is not None
        return self._compute_type

    def transcribe(self, audio_path: Path) -> TranscribeResult:
        self._load()
        assert self._model is not None

        vad_parameters = (
            {"min_silence_duration_ms": self.asr_cfg.vad_min_silence_ms}
            if self.asr_cfg.vad_filter
            else None
        )
        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.model_cfg.language,
            beam_size=self.asr_cfg.beam_size,
            vad_filter=self.asr_cfg.vad_filter,
            vad_parameters=vad_parameters,
            without_timestamps=self.asr_cfg.without_timestamps,
            condition_on_previous_text=self.asr_cfg.condition_on_previous_text,
            repetition_penalty=self.asr_cfg.repetition_penalty,
            no_repeat_ngram_size=self.asr_cfg.no_repeat_ngram_size,
            initial_prompt=self.asr_cfg.initial_prompt or None,
        )

        segments: list[store.Segment] = []
        idx = 0
        for seg in segments_iter:
            text = collapse_repeated_ngrams(seg.text.strip())
            if not text:
                continue
            segments.append(store.Segment(idx=idx, start_s=seg.start, end_s=seg.end, text=text))
            idx += 1

        return TranscribeResult(
            segments=segments,
            device=self._device,  # type: ignore[arg-type]
            compute_type=self._compute_type,  # type: ignore[arg-type]
            language=info.language,
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
