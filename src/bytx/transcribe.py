"""faster-whisper wrapper: loads a converted CTranslate2 Whisper model,
transcribes audio with VAD, and applies anti-hallucination settings.

Handles a known quirk on some GTX 16xx cards where fp16 CUDA kernels
silently produce empty/garbage output: on load we run a 1-second silence
smoke test and fall back down a compute_type chain (and finally to CPU)
if it fails, rather than failing the whole run.

Whisper stops generating after 224 tokens per decode window, and Bengali
needs ~0.7 tokens per character, so a 30 s window overflows and its text is
silently cut off mid-sentence. To avoid that, audio is split at VAD pauses
into short clips (`asr.max_clip_seconds`, overridable per model), each decoded
on its own, and any clip whose output still nears the token limit is split
again and re-decoded.

A Transcriber wraps one model. `TranscriberPool` routes each language to its
configured model and keeps only one model in memory at a time.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from . import glossary as glossary_mod
from . import models, store
from .config import AsrConfig, AsrModelConfig, Config
from .cuda_dll import ensure_cuda_dll_paths

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Whisper decodes at most 224 tokens per window (half its 448 context). A clip
# whose text comes within TOKEN_MARGIN of that was almost certainly cut off.
WHISPER_TOKEN_LIMIT = 224
TOKEN_MARGIN = 8
# Clips shorter than 2x this are never split further.
MIN_CLIP_S = 1.5
# VAD padding can push a lone clip slightly past max_clip_seconds; only clips
# beyond this multiple of it are force-split before decoding.
_MAX_CLIP_SLACK = 1.25
# Language detection transcribes this many clips, spread across the video.
SAMPLE_CLIPS = 10
# Rough tokens-per-character, used only if the model exposes no tokenizer.
_FALLBACK_TOKENS_PER_CHAR = {"bn": 0.75, "en": 0.3}
_DEFAULT_TOKENS_PER_CHAR = 0.5

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


def plan_clips(
    chunks: list[dict], max_clip_s: float, sample_rate: int = SAMPLE_RATE
) -> list[tuple[int, int]]:
    """Greedily merges adjacent VAD speech chunks (dicts with sample-index
    "start"/"end") into clips of at most `max_clip_s` seconds. A clip spans
    from its first chunk's start to its last chunk's end, so the silence
    between merged chunks is kept and timestamps stay real. A single chunk
    already longer than the limit becomes its own clip.
    """
    max_samples = int(max_clip_s * sample_rate)
    clips: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for chunk in chunks:
        start, end = int(chunk["start"]), int(chunk["end"])
        if cur_start is not None and end - cur_start <= max_samples:
            cur_end = end
            continue
        if cur_start is not None:
            clips.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    if cur_start is not None:
        clips.append((cur_start, cur_end))
    return clips


def evenly_spaced(items: list, n: int) -> list:
    """Up to `n` items picked at even intervals, always including the first and
    last (all of them if there are no more than `n`)."""
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    if n == 1:
        return [items[len(items) // 2]]
    picks = dict.fromkeys(round(i * (len(items) - 1) / (n - 1)) for i in range(n))
    return [items[i] for i in picks]


def find_split_point(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_ms: int = 100
) -> int:
    """Returns the sample index of the quietest `frame_ms` frame within the
    middle 40-60% of `audio` -- the least damaging place to cut a clip in two.
    """
    n = len(audio)
    frame = max(1, sample_rate * frame_ms // 1000)
    lo, hi = int(n * 0.4), int(n * 0.6)
    best_idx, best_key = n // 2, (float("inf"), 0)
    for i in range(lo, max(lo + 1, hi - frame + 1), frame):
        window = audio[i : i + frame]
        if len(window) == 0:
            break
        idx = i + len(window) // 2
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        key = (rms, abs(idx - n // 2))  # ties (e.g. uniform audio) prefer the centre
        if key < best_key:
            best_idx, best_key = idx, key
    return best_idx


def _weighted_logprob(segs) -> float | None:
    """Mean `avg_logprob` of faster-whisper segments, weighted by their length
    (equal weights when none has any, e.g. timestamps are off)."""
    if not segs:
        return None
    weights = [max(seg.end - seg.start, 0.0) for seg in segs]
    if sum(weights) <= 0:
        weights = [1.0] * len(segs)
    return sum(seg.avg_logprob * w for seg, w in zip(segs, weights, strict=True)) / sum(weights)


@dataclass
class TranscribeStats:
    """Diagnostics from one transcription run, used to flag suspect output."""

    clips: int = 0
    resplit: int = 0  # clips split and re-decoded because they hit the token limit
    truncated_unresolved: int = 0  # clips still at the limit at minimum size
    speech_seconds: float = 0.0
    audio_seconds: float = 0.0


@dataclass
class _Piece:
    """Text decoded from audio[start:end] (sample indices), with the model's
    confidence in it (None when the decode produced no scored segment)."""

    start: int
    end: int
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class TranscribeResult:
    segments: list[store.Segment]
    device: str
    compute_type: str
    language: str | None
    stats: TranscribeStats = field(default_factory=TranscribeStats)


class ModelLoadError(RuntimeError):
    pass


def _physical_cores() -> int:
    return psutil.cpu_count(logical=False) or os.cpu_count() or 4


class Transcriber:
    """Lazily loads the model on first use; call `unload()` between the ASR
    stage and later GPU-using stages (e.g. Ollama) to free VRAM.
    """

    def __init__(
        self,
        model_dir: Path,
        model_cfg: AsrModelConfig,
        asr_cfg: AsrConfig,
        glossary: glossary_mod.Glossary | None = None,
    ):
        self.model_dir = model_dir
        self.model_cfg = model_cfg
        self.asr_cfg = asr_cfg
        self.glossary = glossary
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
                f"Model directory {self.model_dir} does not exist. Run `bytx setup` first."
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
                # CTranslate2 uses 4 CPU threads unless told; one per physical core is faster.
                threads = {"cpu_threads": _physical_cores()} if device == "cpu" else {}
                try:
                    model = WhisperModel(
                        str(self.model_dir), device=device, compute_type=compute_type, **threads
                    )
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
        segments, _info = model.transcribe(
            silence, language=self.model_cfg.language or "en", beam_size=1
        )
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

    @property
    def max_clip_seconds(self) -> float:
        return self.model_cfg.max_clip_seconds or self.asr_cfg.max_clip_seconds

    def _resolve_language(self, language: str | None) -> str:
        """A pinned model always decodes in its own language; a multilingual
        model needs the caller to say which."""
        pinned = self.model_cfg.language
        if pinned:
            if language and language != pinned:
                raise ValueError(
                    f"Model {self.model_cfg.key!r} is pinned to {pinned!r} and cannot "
                    f"transcribe {language!r}"
                )
            return pinned
        if not language:
            raise ValueError(
                f"Model {self.model_cfg.key!r} is multilingual; pass the language to transcribe"
            )
        return language

    def _vad_options(self) -> VadOptions:
        return VadOptions(
            threshold=self.asr_cfg.vad_threshold,
            min_silence_duration_ms=self.asr_cfg.vad_min_silence_ms,
            max_speech_duration_s=self.max_clip_seconds,
            speech_pad_ms=self.asr_cfg.vad_speech_pad_ms,
        )

    def _plan_clips(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Speech in `audio` as (start, end) sample ranges of at most max_clip_seconds."""
        if self.asr_cfg.vad_filter:
            chunks = get_speech_timestamps(audio, self._vad_options())
        else:
            chunks = [{"start": 0, "end": len(audio)}] if len(audio) else []
        return plan_clips(chunks, self.max_clip_seconds)

    def sample_text(self, audio_path: Path, language: str | None = None, clips: int = SAMPLE_CLIPS) -> str:
        """Transcribes about `clips` speech clips spread evenly across the audio
        (so an intro in another language doesn't decide it) and returns the text.
        A cheap look at what this model makes of the audio, for language detection.
        """
        language = self._resolve_language(language)
        self._load()
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        stats = TranscribeStats()
        pieces: list[_Piece] = []
        for start, end in evenly_spaced(self._plan_clips(audio), clips):
            self._decode_range(audio, start, end, stats, pieces, language)
        return " ".join(p.text for p in pieces)

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscribeResult:
        language = self._resolve_language(language)
        self._load()
        assert self._model is not None

        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        clips = self._plan_clips(audio)

        stats = TranscribeStats(
            clips=len(clips),
            speech_seconds=sum(e - s for s, e in clips) / SAMPLE_RATE,
            audio_seconds=len(audio) / SAMPLE_RATE,
        )
        pieces: list[_Piece] = []
        for start, end in clips:
            self._decode_range(audio, start, end, stats, pieces, language)

        segments = [
            store.Segment(
                idx=i, start_s=p.start / SAMPLE_RATE, end_s=p.end / SAMPLE_RATE, text=p.text,
                avg_logprob=p.avg_logprob, no_speech_prob=p.no_speech_prob,
            )
            for i, p in enumerate(pieces)
        ]
        return TranscribeResult(
            segments=segments,
            device=self._device,  # type: ignore[arg-type]
            compute_type=self._compute_type,  # type: ignore[arg-type]
            language=language,
            stats=stats,
        )

    def _decode_range(
        self,
        audio: np.ndarray,
        start: int,
        end: int,
        stats: TranscribeStats,
        out: list[_Piece],
        language: str,
    ) -> None:
        """Decodes audio[start:end] into `out` as pieces, in
        time order. Splits the range in two (at its quietest point) when it is
        too long, or when the decoded text nears Whisper's token limit and so
        was probably cut off.
        """
        duration_s = (end - start) / SAMPLE_RATE
        can_split = duration_s >= 2 * MIN_CLIP_S

        if can_split and duration_s > self.max_clip_seconds * _MAX_CLIP_SLACK:
            mid = start + find_split_point(audio[start:end])
            self._decode_range(audio, start, mid, stats, out, language)
            self._decode_range(audio, mid, end, stats, out, language)
            return

        piece = self._decode_clip(audio[start:end], language)
        piece.start, piece.end = start, end
        text = piece.text
        if text and self._nears_token_limit(text, language):
            if can_split:
                stats.resplit += 1
                mid = start + find_split_point(audio[start:end])
                self._decode_range(audio, start, mid, stats, out, language)
                self._decode_range(audio, mid, end, stats, out, language)
                return
            stats.truncated_unresolved += 1
        if text:
            out.append(piece)

    def _initial_prompt(self, language: str) -> str | None:
        """The configured prompt plus, for languages that opt in, the glossary's terms."""
        parts = [self.asr_cfg.initial_prompt]
        if self.glossary and language in self.asr_cfg.glossary_prompt_languages:
            parts.append(self.glossary.prompt(self.asr_cfg.glossary_prompt_max_chars))
        return ", ".join(p for p in parts if p) or None

    def _decode_clip(self, clip: np.ndarray, language: str) -> _Piece:
        """Decodes one clip. The returned piece has no start/end yet (the caller
        knows where the clip sits); its confidence is the duration-weighted mean
        log-probability, and the highest no-speech probability, of the model's
        segments for it."""
        assert self._model is not None
        segments_iter, _info = self._model.transcribe(
            clip,
            language=language,
            beam_size=self.asr_cfg.beam_size,
            vad_filter=False,
            without_timestamps=self.asr_cfg.without_timestamps,
            condition_on_previous_text=self.asr_cfg.condition_on_previous_text,
            repetition_penalty=self.asr_cfg.repetition_penalty,
            no_repeat_ngram_size=self.asr_cfg.no_repeat_ngram_size,
            initial_prompt=self._initial_prompt(language),
        )
        segs = list(segments_iter)
        text = collapse_repeated_ngrams(" ".join(seg.text.strip() for seg in segs).strip())
        return _Piece(
            start=0, end=0, text=text,
            avg_logprob=_weighted_logprob(segs),
            no_speech_prob=max((seg.no_speech_prob for seg in segs), default=None),
        )

    def _nears_token_limit(self, text: str, language: str) -> bool:
        tokenizer = getattr(self._model, "hf_tokenizer", None)
        if tokenizer is not None:
            n_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
        else:
            per_char = _FALLBACK_TOKENS_PER_CHAR.get(language, _DEFAULT_TOKENS_PER_CHAR)
            n_tokens = int(len(text) * per_char)
        return n_tokens >= WHISPER_TOKEN_LIMIT - TOKEN_MARGIN

    def unload(self) -> None:
        self._model = None
        gc.collect()


class TranscriberPool:
    """Hands out the Transcriber for a language and keeps at most one model
    loaded: switching to a different model unloads the previous one first, so
    two models never compete for VRAM.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._current: Transcriber | None = None
        self._glossary = glossary_mod.load(cfg.asr.glossary)

    def _get(self, model_cfg: AsrModelConfig) -> Transcriber:
        if self._current is not None and self._current.model_cfg.key == model_cfg.key:
            return self._current
        self.unload()
        self._current = Transcriber(
            models.local_path(model_cfg), model_cfg, self.cfg.asr, self._glossary
        )
        return self._current

    def for_language(self, language: str) -> Transcriber:
        return self._get(self.cfg.asr.model_for_language(language))

    def unload(self) -> None:
        if self._current is not None:
            self._current.unload()
            self._current = None
