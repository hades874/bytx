from __future__ import annotations

import huggingface_hub
import pytest

from bytx import models
from bytx.config import AsrModelConfig, load_config


def model_cfg(source, already_ct2=True, key="m"):
    return AsrModelConfig(key=key, source=str(source), language="", already_ct2=already_ct2)


def ct2_folder(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(b"")
    return path


@pytest.fixture()
def hf_cache(tmp_path, monkeypatch):
    """A fake Hugging Face cache: repo id -> snapshot folder. Records downloads."""
    cache: dict[str, object] = {}
    downloads: list[tuple[str, bool]] = []

    def snapshot_download(repo_id, local_files_only=False, force_download=False):
        if local_files_only:
            if repo_id not in cache:
                raise FileNotFoundError(f"{repo_id} is not cached")  # what huggingface_hub raises
            return str(cache[repo_id])
        downloads.append((repo_id, force_download))
        cache[repo_id] = ct2_folder(tmp_path / "hf" / repo_id.replace("/", "--"))
        return str(cache[repo_id])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return cache, downloads


# --- local_path ---------------------------------------------------------------------

def test_a_local_folder_source_is_used_as_it_is(tmp_path, hf_cache):
    folder = ct2_folder(tmp_path / "my-model")
    assert models.local_path(model_cfg(folder, already_ct2=False)) == folder


def test_a_local_folder_without_a_model_is_reported(tmp_path, hf_cache):
    (tmp_path / "empty").mkdir()
    with pytest.raises(models.ModelsMissing, match="holds no model.bin"):
        models.local_path(model_cfg(tmp_path / "empty"))


def test_a_relative_source_is_never_looked_up_in_the_current_folder(tmp_path, hf_cache, monkeypatch):
    ct2_folder(tmp_path / "org" / "repo")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(models.ModelsMissing, match="is not downloaded"):
        models.local_path(model_cfg("org/repo"))


def test_a_folder_source_that_is_not_there_says_to_fix_it(tmp_path, hf_cache):
    nowhere = model_cfg(tmp_path / "nowhere", already_ct2=False)
    for find in (models.local_path, models.download):
        with pytest.raises(models.ModelsMissing, match="is not a folder on this machine"):
            find(nowhere)


def test_a_cached_repo_resolves_to_its_snapshot(tmp_path, hf_cache):
    cache, downloads = hf_cache
    cache["org/m"] = ct2_folder(tmp_path / "snap")
    assert models.local_path(model_cfg("org/m")) == tmp_path / "snap"
    assert downloads == []  # never downloads


def test_an_uncached_repo_says_run_setup(hf_cache):
    with pytest.raises(models.ModelsMissing, match="Run `bytx setup`"):
        models.local_path(model_cfg("org/m"))


def test_a_partly_downloaded_snapshot_counts_as_missing(tmp_path, hf_cache):
    cache, _ = hf_cache
    (tmp_path / "partial").mkdir()
    cache["org/m"] = tmp_path / "partial"  # no model.bin yet
    with pytest.raises(models.ModelsMissing):
        models.local_path(model_cfg("org/m"))


# --- download -------------------------------------------------------------------------

def test_download_fetches_a_missing_model(hf_cache):
    _, downloads = hf_cache
    path = models.download(model_cfg("org/m"))
    assert (path / "model.bin").is_file()
    assert downloads == [("org/m", False)]


def test_download_skips_a_model_that_is_already_there(hf_cache):
    _, downloads = hf_cache
    models.download(model_cfg("org/m"))
    models.download(model_cfg("org/m"))
    assert downloads == [("org/m", False)]


def test_download_with_force_fetches_again(hf_cache):
    _, downloads = hf_cache
    models.download(model_cfg("org/m"))
    models.download(model_cfg("org/m"), force=True)
    assert downloads == [("org/m", False), ("org/m", True)]


def test_download_refuses_a_model_that_needs_converting(hf_cache):
    _, downloads = hf_cache
    with pytest.raises(ValueError, match="scripts/convert_model.py"):
        models.download(model_cfg("org/raw-whisper", already_ct2=False))
    assert downloads == []


def test_download_of_a_local_folder_downloads_nothing(tmp_path, hf_cache):
    _, downloads = hf_cache
    folder = ct2_folder(tmp_path / "mine")
    assert models.download(model_cfg(folder, already_ct2=False)) == folder
    assert downloads == []


# --- missing --------------------------------------------------------------------------

def test_missing_lists_only_the_configured_models_that_are_absent(tmp_path, hf_cache):
    cache, _ = hf_cache
    cache["org/bn"] = ct2_folder(tmp_path / "bn")
    (tmp_path / "config.toml").write_text(
        '[asr]\nmodel = "bn"\n[asr.language_models]\nbn = "bn"\nen = "multi"\n'
        '[asr.models.bn]\nsource = "org/bn"\nlanguage = "bn"\n'
        '[asr.models.multi]\nsource = "org/multi"\nlanguage = ""\n',
        encoding="utf-8",
    )
    assert models.missing(load_config(tmp_path / "config.toml")) == ["multi"]


# --- download_size ----------------------------------------------------------------------

class _File:
    def __init__(self, size):
        self.size = size


class _Info:
    siblings = [_File(1_527_906_453), _File(339), _File(None)]


def test_download_size_adds_up_the_repo_files(monkeypatch):
    calls = []
    monkeypatch.setattr(
        huggingface_hub.HfApi, "model_info",
        lambda self, repo, files_metadata=False: calls.append((repo, files_metadata)) or _Info(),
    )
    assert models.download_size(model_cfg("org/m")) == 1_527_906_792
    assert calls == [("org/m", True)]


def test_download_size_is_unknown_offline(monkeypatch):
    def offline(self, repo, files_metadata=False):
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", offline)
    assert models.download_size(model_cfg("org/m")) is None


def test_download_size_never_asks_the_hub_about_local_or_unconvertible_models(tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", lambda *a, **k: pytest.fail("asked the Hub"))
    assert models.download_size(model_cfg(ct2_folder(tmp_path / "local"))) is None
    assert models.download_size(model_cfg("org/raw", already_ct2=False)) is None
