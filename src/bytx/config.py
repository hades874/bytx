"""Loads config.toml into typed, dot-accessible config objects.

Kept deliberately dependency-free (stdlib tomllib only) so this module can be
imported early, before any heavy ML libraries.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_FILENAME = "config.toml"


@dataclass(frozen=True)
class AsrModelConfig:
    key: str
    source: str
    language: str
    already_ct2: bool = False


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

    def active_model(self) -> AsrModelConfig:
        try:
            return self.models[self.model]
        except KeyError as exc:
            known = ", ".join(sorted(self.models)) or "(none defined)"
            raise ValueError(
                f"asr.model = {self.model!r} is not defined under [asr.models.*]. "
                f"Known models: {known}"
            ) from exc


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

    def ensure_dirs(self) -> None:
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.videos_dir.mkdir(parents=True, exist_ok=True)
        self.paths.audio_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.transcribed_files_dir.mkdir(parents=True, exist_ok=True)


def _resolve(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def load_config(path: str | Path | None = None) -> Config:
    """Load config.toml. Defaults to config.toml next to the project root
    (found by walking up from cwd looking for pyproject.toml), falling back
    to the current directory.
    """
    if path is not None:
        config_path = Path(path)
        root = config_path.parent
    else:
        root = _find_project_root(Path.cwd())
        config_path = root / DEFAULT_CONFIG_FILENAME

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Copy config.toml from the project root or pass --config."
        )

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

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
            language=m.get("language", "bn"),
            already_ct2=m.get("already_ct2", False),
        )
        for key, m in models_raw.items()
    }
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
        language=summarize_raw.get("language", "bn"),
    )

    return Config(
        root=root,
        paths=paths,
        asr=asr,
        fetch=fetch,
        ollama=ollama,
        rag=rag,
        summarize=summarize,
    )


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start
