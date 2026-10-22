from __future__ import annotations

import importlib.util

from bytx import cuda_dll


def test_an_install_without_the_gpu_extra_adds_nothing(monkeypatch):
    def no_nvidia_package(name):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(cuda_dll.sys, "platform", "win32")
    monkeypatch.setattr(importlib.util, "find_spec", no_nvidia_package)
    assert cuda_dll.ensure_cuda_dll_paths() == []


def test_other_platforms_are_left_alone(monkeypatch):
    monkeypatch.setattr(cuda_dll.sys, "platform", "linux")
    assert cuda_dll.ensure_cuda_dll_paths() == []
