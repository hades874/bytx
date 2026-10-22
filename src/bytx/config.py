"""Loads the settings into typed, dot-accessible config objects: the packaged
defaults.toml with the user's config.toml merged over it.

Kept deliberately dependency-free (stdlib tomllib only) so this module can be
imported early, before any heavy ML libraries.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from . import langs

DEFAULT_CONFIG_FILENAME = "config.toml"
DEFAULTS_FILENAME = "defaults.toml"


@dataclass(frozen=True)
class AsrModelConfig:
    key: str
    source: str
    # Language this model is pinned to (e.g. a Bangla-only fine-tune). Empty
    # means multilingual: the language is chosen per video.
    language: str
    already_ct2: bool = False
    # Overrides asr.max_clip_seconds for this model (its language's token
    # density decides how long a clip can be before Whisper's 224-token cap).
    max_clip_seconds: float | None = None

    @property
    def multilingual(self) -> bool:
        return not self.language


@dataclass(frozen=True)
class AsrConfig:
    model: str
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    vad_min_silence_ms: int
    without_timestamps: bool
    condition_on_previous_text: bool
    repetition_penalty: float
    no_repeat_ngram_size: int
    initial_prompt: str
    models: dict[str, AsrModelConfig] = field(default_factory=dict)
    # Decode-window sizing. Whisper stops after 224 tokens per window, which
    # Bengali overflows in well under 30 s, so audio is cut into short clips.
    max_clip_seconds: float = 12.0
    vad_threshold: float = 0.5
    vad_speech_pad_ms: int = 400
    # Language code -> model key (see [asr.language_models]). When empty, the
    # single `model` serves every language it is able to.
    language_models: dict[str, str] = field(default_factory=dict)
    # Language of videos added without `--lang`.
    default_language: str = langs.BANGLA
    # `--lang auto` calls a video English when fewer than this share of the
    # words the Bangla model writes for a sample of it are Bangla function
    # words (see langdetect.py).
    auto_english_below: float = 0.10
    # Vocabulary file (see glossary.py); relative to the project root. None disables it.
    glossary: Path | None = None
    # Languages whose decoding is primed with the glossary's terms. Priming a
    # fine-tuned model can make it echo the prompt, so Bangla is opt-in.
    glossary_prompt_languages: tuple[str, ...] = (langs.ENGLISH,)
    glossary_prompt_max_chars: int = 300

    def _model(self, key: str) -> AsrModelConfig:
        try:
            return self.models[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.models)) or "(none defined)"
            raise ValueError(
                f"ASR model {key!r} is not defined under [asr.models.*]. Known models: {known}"
            ) from exc

    def active_model(self) -> AsrModelConfig:
        return self._model(self.model)

    def model_for_language(self, language: str) -> AsrModelConfig:
        """The model that transcribes `language`: the [asr.language_models]
        entry, else `asr.model` if it can handle that language."""
        key = self.language_models.get(language)
        if key is not None:
            return self._model(key)
        active = self.active_model()
        if active.multilingual or active.language == language:
            return active
        configured = ", ".join(sorted(self.language_models)) or active.language
        raise ValueError(
            f"No ASR model is configured for language {language!r} "
            f"(configured: {configured}). Add it under [asr.language_models]."
        )

    def required_model_keys(self) -> list[str]:
        """Every model this config can route to, deduplicated, for setup/doctor."""
        return list(dict.fromkeys([self.model, *self.language_models.values()]))


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for flagging a transcript as suspect (see quality.py)."""

    min_wpm: float = 70.0  # fewer words per minute than this: speech was probably dropped
    max_wpm: float = 220.0  # more than this: probably repeated or hallucinated text
    max_gap_seconds: float = 10.0  # a longer stretch with no text (also at the start/end)
    min_avg_logprob: float = -1.0  # a clip whose mean token log-probability is below this
    max_no_speech_prob: float = 0.6  # a clip the model thinks may hold no speech


@dataclass(frozen=True)
class FetchConfig:
    force_asr: bool
    sleep_interval_s: float
    max_sleep_interval_s: float
    cookies_from_browser: str


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    embed_model: str
    chat_model: str
    num_ctx: int


@dataclass(frozen=True)
class RagConfig:
    chunk_seconds: int
    chunk_overlap_seconds: int
    top_k: int


@dataclass(frozen=True)
class SummarizeConfig:
    enabled: bool
    map_window_minutes: int
    # "auto" writes each summary in its video's language; "bn"/"en" force one.
    language: str


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    models_dir: Path
    transcribed_files_dir: Path

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "library.db"


@dataclass(frozen=True)
class Config:
    root: Path
    paths: PathsConfig
    asr: AsrConfig
    fetch: FetchConfig
    ollama: OllamaConfig
    rag: RagConfig
    summarize: SummarizeConfig
    quality: QualityConfig = field(default_factory=QualityConfig)

    def ensure_dirs(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.videos_dir.mkdir(parents=True, exist_ok=True)
        self.paths.audio_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.transcribed_files_dir.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _merge(base: dict, over: dict) -> dict:
    """`over` on top of `base`: tables merge key by key, any other value replaces."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_defaults() -> dict:
    """The packaged defaults.toml, as raw TOML data."""
    return tomllib.loads((files(__package__) / DEFAULTS_FILENAME).read_text(encoding="utf-8"))


def load_config(path: str | Path | None = None) -> Config:
    """Load the packaged defaults with the user's config.toml merged over them.
    `path` names the user file (it must exist); otherwise it is config.toml
    next to the project root (found by walking up from cwd looking for
    pyproject.toml), and may be missing.
    """
    if path is not None:
        config_path = Path(path)
        root = config_path.parent
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}.")
    else:
        root = _find_project_root(Path.cwd())
        config_path = root / DEFAULT_CONFIG_FILENAME

    raw = load_defaults()
    if config_path.exists():
        with config_path.open("rb") as f:
            raw = _merge(raw, tomllib.load(f))

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        data_dir=_resolve(root, paths_raw.get("data_dir", "data")),
        models_dir=_resolve(root, paths_raw.get("models_dir", "models")),
        transcribed_files_dir=_resolve(
            root, paths_raw.get("transcribed_files_dir", "yt-transcribed-files")
        ),
    )

    asr_raw = raw.get("asr", {})
    models_raw = asr_raw.get("models", {})
    models = {
        key: AsrModelConfig(
            key=key,
            source=m["source"],
            language=m.get("language", langs.BANGLA),
            already_ct2=m.get("already_ct2", False),
            max_clip_seconds=m.get("max_clip_seconds"),
        )
        for key, m in models_raw.items()
    }
    language_models = dict(asr_raw.get("language_models", {}))
    for lang, key in language_models.items():
        if lang not in langs.SUPPORTED:
            raise ValueError(
                f"[asr.language_models] has unsupported language {lang!r} "
                f"(supported: {', '.join(langs.SUPPORTED)})"
            )
        if key not in models:
            raise ValueError(
                f"[asr.language_models] {lang} = {key!r} is not defined under [asr.models.*]"
            )
        if models[key].language and models[key].language != lang:
            raise ValueError(
                f"[asr.language_models] {lang} = {key!r}, but that model is pinned to "
                f"{models[key].language!r}"
            )
    default_language = asr_raw.get("default_language", langs.BANGLA)
    if default_language not in langs.SUPPORTED:
        raise ValueError(
            f"asr.default_language = {default_language!r} is not supported "
            f"(supported: {', '.join(langs.SUPPORTED)})"
        )
    prompt_languages = tuple(asr_raw.get("glossary_prompt_languages", [langs.ENGLISH]))
    for lang in prompt_languages:
        if lang not in langs.SUPPORTED:
            raise ValueError(
                f"asr.glossary_prompt_languages has unsupported language {lang!r} "
                f"(supported: {', '.join(langs.SUPPORTED)})"
            )
    glossary_raw = asr_raw.get("glossary", "glossary.txt")
    asr = AsrConfig(
        model=asr_raw.get("model", "tugstugi-medium-bn"),
        device=asr_raw.get("device", "auto"),
        compute_type=asr_raw.get("compute_type", "float16"),
        beam_size=asr_raw.get("beam_size", 1),
        vad_filter=asr_raw.get("vad_filter", True),
        vad_min_silence_ms=asr_raw.get("vad_min_silence_ms", 500),
        without_timestamps=asr_raw.get("without_timestamps", True),
        condition_on_previous_text=asr_raw.get("condition_on_previous_text", False),
        repetition_penalty=asr_raw.get("repetition_penalty", 1.0),
        no_repeat_ngram_size=asr_raw.get("no_repeat_ngram_size", 0),
        initial_prompt=asr_raw.get("initial_prompt", ""),
        models=models,
        max_clip_seconds=asr_raw.get("max_clip_seconds", 12.0),
        vad_threshold=asr_raw.get("vad_threshold", 0.5),
        vad_speech_pad_ms=asr_raw.get("vad_speech_pad_ms", 400),
        language_models=language_models,
        default_language=default_language,
        auto_english_below=asr_raw.get("auto_english_below", 0.10),
        glossary=_resolve(root, glossary_raw) if glossary_raw else None,
        glossary_prompt_languages=prompt_languages,
        glossary_prompt_max_chars=asr_raw.get("glossary_prompt_max_chars", 300),
    )

    fetch_raw = raw.get("fetch", {})
    fetch = FetchConfig(
        force_asr=fetch_raw.get("force_asr", False),
        sleep_interval_s=fetch_raw.get("sleep_interval_s", 1),
        max_sleep_interval_s=fetch_raw.get("max_sleep_interval_s", 5),
        cookies_from_browser=fetch_raw.get("cookies_from_browser", ""),
    )

    ollama_raw = raw.get("ollama", {})
    ollama = OllamaConfig(
        host=ollama_raw.get("host", "http://127.0.0.1:11434"),
        embed_model=ollama_raw.get("embed_model", "bge-m3"),
        chat_model=ollama_raw.get("chat_model", "gemma3:4b"),
        num_ctx=ollama_raw.get("num_ctx", 8192),
    )

    rag_raw = raw.get("rag", {})
    rag = RagConfig(
        chunk_seconds=rag_raw.get("chunk_seconds", 75),
        chunk_overlap_seconds=rag_raw.get("chunk_overlap_seconds", 15),
        top_k=rag_raw.get("top_k", 8),
    )

    summarize_raw = raw.get("summarize", {})
    summarize = SummarizeConfig(
        enabled=summarize_raw.get("enabled", True),
        map_window_minutes=summarize_raw.get("map_window_minutes", 10),
        language=summarize_raw.get("language", "auto"),
    )

    quality_raw = raw.get("quality", {})
    defaults = QualityConfig()
    quality = QualityConfig(
        min_wpm=quality_raw.get("min_wpm", defaults.min_wpm),
        max_wpm=quality_raw.get("max_wpm", defaults.max_wpm),
        max_gap_seconds=quality_raw.get("max_gap_seconds", defaults.max_gap_seconds),
        min_avg_logprob=quality_raw.get("min_avg_logprob", defaults.min_avg_logprob),
        max_no_speech_prob=quality_raw.get("max_no_speech_prob", defaults.max_no_speech_prob),
    )

    return Config(
        root=root,
        paths=paths,
        asr=asr,
        fetch=fetch,
        ollama=ollama,
        rag=rag,
        summarize=summarize,
        quality=quality,
    )


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start
