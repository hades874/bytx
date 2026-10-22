"""Where the ASR models are on this machine, and getting them there.

Models live in the Hugging Face cache (move it with HF_HOME), not in the data
folder. A model's `source` is either a Hugging Face repo holding a CTranslate2
conversion (`already_ct2 = true`), which `bytx setup` downloads, or a local
folder holding one, which is used as it is. Converting any other checkpoint is
a maintainer job (scripts/convert_model.py), never part of setup.
"""

from __future__ import annotations

from pathlib import Path

from .config import AsrModelConfig, Config

MARKER = "model.bin"  # every CTranslate2 model folder has one


class ModelsMissing(RuntimeError):
    """An ASR model the config needs is not on this machine."""


def _local_folder(model_cfg: AsrModelConfig) -> Path | None:
    folder = Path(model_cfg.source).expanduser()
    return folder if folder.is_dir() else None


def local_path(model_cfg: AsrModelConfig) -> Path:
    """The folder to load `model_cfg` from. Never downloads anything; raises
    ModelsMissing when the model is not there."""
    folder = _local_folder(model_cfg)
    if folder is not None:
        if not (folder / MARKER).is_file():
            raise ModelsMissing(
                f"ASR model {model_cfg.key!r}: {folder} holds no {MARKER}; "
                "its `source` must be a CTranslate2 model folder."
            )
        return folder

    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(model_cfg.source, local_files_only=True))
    except FileNotFoundError:  # huggingface_hub's LocalEntryNotFoundError: not cached
        path = None
    if path is None or not (path / MARKER).is_file():
        raise ModelsMissing(
            f"ASR model {model_cfg.key!r} ({model_cfg.source}) is not downloaded. Run `bytx setup`."
        )
    return path


def download(model_cfg: AsrModelConfig, force: bool = False) -> Path:
    """Makes `model_cfg` available locally and returns its folder. Skips the
    download when it is already there, unless `force`."""
    if _local_folder(model_cfg) is not None:
        return local_path(model_cfg)
    if not model_cfg.already_ct2:
        raise ValueError(
            f"ASR model {model_cfg.key!r} ({model_cfg.source}) is not a CTranslate2 model, so "
            "`bytx setup` can't download it. Convert it with scripts/convert_model.py (needs the "
            "'convert' extra) and set its `source` to the output folder."
        )
    if not force:
        try:
            return local_path(model_cfg)
        except ModelsMissing:
            pass

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_cfg.source, force_download=force))


def download_size(model_cfg: AsrModelConfig) -> int | None:
    """Bytes `download` would fetch from Hugging Face; None for a local folder,
    a model setup can't download, or when the Hub can't be asked (offline)."""
    if _local_folder(model_cfg) is not None or not model_cfg.already_ct2:
        return None
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(model_cfg.source, files_metadata=True)
    except Exception:  # noqa: BLE001 - a size is a nicety; the download reports real errors
        return None
    return sum(f.size or 0 for f in info.siblings or [])


def missing(cfg: Config) -> list[str]:
    """Keys of the models `bytx setup` would fetch that are not on this machine."""
    absent = []
    for key in cfg.asr.required_model_keys():
        model_cfg = cfg.asr.models.get(key)
        if model_cfg is None:
            continue  # an undefined asr.model only matters to `bench`, which reports it
        try:
            local_path(model_cfg)
        except ModelsMissing:
            absent.append(key)
    return absent
