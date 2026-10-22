#!/usr/bin/env python
"""Maintainer tool: convert a Hugging Face Whisper checkpoint to CTranslate2
format for faster-whisper.

bytx itself never converts: `bytx setup` only downloads models that are
already CTranslate2 (`already_ct2 = true`). This script is how such a model
is made, e.g. the Bangla model published as bytx/tugstugi-medium-bn-ct2
(see docs/hf/). It needs the 'convert' extra (transformers + torch):

    uv sync --extra convert
    uv run --extra convert python scripts/convert_model.py \\
        bengaliAI/tugstugi_bengaliai-asr_whisper-medium out/tugstugi-medium-bn

To use a converted model without publishing it, set that model's `source`
in your config.toml to the output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert(source: str, out_dir: Path, quantization: str, force: bool) -> Path:
    if (out_dir / "model.bin").exists() and not force:
        print(f"{out_dir} already holds a converted model. Use --force to redo.")
        return out_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {source} -> {out_dir} (quantization={quantization})")
    from ctranslate2.converters.transformers import TransformersConverter

    converter = TransformersConverter(source, copy_files=["tokenizer.json", "preprocessor_config.json"])
    converter.convert(str(out_dir), quantization=quantization, force=force)
    print(f"Done: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="Hugging Face repo id (or local folder) of a Whisper checkpoint")
    parser.add_argument("out_dir", type=Path, help="Folder to write the CTranslate2 model to")
    parser.add_argument(
        "--quantization", default="float16", help="CTranslate2 quantization (default: float16, as published)"
    )
    parser.add_argument("--force", action="store_true", help="Convert again even if out_dir has a model")
    args = parser.parse_args()
    convert(args.source, args.out_dir, args.quantization, args.force)


if __name__ == "__main__":
    main()
