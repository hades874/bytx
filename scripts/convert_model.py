#!/usr/bin/env python
"""One-off conversion of a HuggingFace Whisper checkpoint into CTranslate2
format for use with faster-whisper, or a download of an already-converted
CTranslate2 model repo.

Requires the 'convert' extra (transformers + torch), which the normal
runtime does not need:

    uv sync --extra convert
    uv run --extra convert python scripts/convert_model.py <model-key>

`<model-key>` must match a key under [asr.models.*] in config.toml.
Normally you don't call this directly -- `bytx setup` runs it for the
configured model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bytx.config import load_config  # noqa: E402


def convert_or_download(
    model_key: str, config_path: str | None, quantization: str | None, force: bool
) -> Path:
    cfg = load_config(config_path)
    try:
        model_cfg = cfg.asr.models[model_key]
    except KeyError as exc:
        known = ", ".join(sorted(cfg.asr.models)) or "(none defined)"
        raise SystemExit(f"Unknown model key {model_key!r}. Known: {known}") from exc

    out_dir = cfg.paths.models_dir / model_key
    marker = out_dir / "model.bin"

    if model_cfg.already_ct2:
        if marker.exists() and not force:
            print(f"[bytx] {model_key} already present at {out_dir}")
            return out_dir
        print(f"[bytx] Downloading pre-converted CTranslate2 model {model_cfg.source} -> {out_dir}")
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=model_cfg.source, local_dir=str(out_dir))
        print(f"[bytx] Done: {out_dir}")
        return out_dir

    if marker.exists() and not force:
        print(f"[bytx] {model_key} already converted at {out_dir}. Use --force to redo.")
        return out_dir

    quantization = quantization or cfg.asr.compute_type
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bytx] Converting {model_cfg.source} -> {out_dir} (quantization={quantization})")
    from ctranslate2.converters.transformers import TransformersConverter

    converter = TransformersConverter(
        model_cfg.source,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    converter.convert(str(out_dir), quantization=quantization, force=force)
    print(f"[bytx] Done: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_key", help="Key under [asr.models.*] in config.toml")
    parser.add_argument(
        "--quantization", default=None, help="Override the quantization/compute_type for this conversion"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-convert/re-download even if the output already exists"
    )
    parser.add_argument("--config", default=None, help="Path to config.toml")
    args = parser.parse_args()
    convert_or_download(args.model_key, args.config, args.quantization, args.force)


if __name__ == "__main__":
    main()
