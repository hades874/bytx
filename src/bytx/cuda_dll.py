"""Fixes a known Windows issue where pip-installed NVIDIA CUDA/cuDNN wheels
(nvidia-cublas-cu12, nvidia-cudnn-cu12, from the `gpu` extra) drop their DLLs under
site-packages/nvidia/<pkg>/bin, a directory Windows never adds to the DLL
search path on its own. CTranslate2 (used by faster-whisper) then silently
falls back to CPU instead of raising an error.

Call `ensure_cuda_dll_paths()` once, before constructing a WhisperModel,
whenever device="cuda" (or "auto") is requested on Windows.
"""

from __future__ import annotations

import importlib.util
import os
import sys


def ensure_cuda_dll_paths() -> list[str]:
    """Adds the bin/ directories of pip-installed nvidia-*-cu12 packages to
    the DLL search path. Returns the list of directories that were added
    (empty on non-Windows, or if the packages aren't installed).

    Two mechanisms are used together, because they cover different loaders:
    - `os.add_dll_directory()` registers the dir with the Win32
      AddDllDirectory API, which only helps callers that load via
      LoadLibraryEx with LOAD_LIBRARY_SEARCH_* flags (e.g. Python's own
      ctypes/import machinery).
    - Prepending to the `PATH` env var additionally covers native code
      (such as CTranslate2's own DLL-loading shim) that calls the plain
      LoadLibraryW, which ignores AddDllDirectory-registered paths but
      still searches PATH. Empirically, CTranslate2 needs this: without
      it, `cublas64_12.dll is not found or cannot be loaded` is raised
      from inside `WhisperModel.transcribe()` even though the file exists
      and add_dll_directory alone succeeds for a direct ctypes load.
    """
    if sys.platform != "win32":
        return []

    added: list[str] = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:  # no `nvidia` package at all: an install without the gpu extra
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for location in spec.submodule_search_locations:
            bin_dir = os.path.join(location, "bin")
            if os.path.isdir(bin_dir):
                try:
                    os.add_dll_directory(bin_dir)
                except (OSError, AttributeError):
                    pass
                added.append(bin_dir)

    if added:
        current_path = os.environ.get("PATH", "")
        new_dirs = [d for d in added if d not in current_path]
        if new_dirs:
            os.environ["PATH"] = os.pathsep.join([*new_dirs, current_path])

    return added
