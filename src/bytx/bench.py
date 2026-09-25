"""`bytx bench`: compares ASR models on a real clip -- speed (RTF) and,
given a reference transcript, WER/CER via jiwer.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from rich.markup import escape

from . import fetch
from .config import Config
from .transcribe import Transcriber


def parse_timestamp(ts: str) -> float:
    """Parses 'ss', 'mm:ss', or 'hh:mm:ss' into seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_clip(clip: str) -> tuple[float, float]:
    """Parses 'mm:ss-mm:ss' (or hh:mm:ss variants) into (start_s, end_s)."""
    start_str, end_str = clip.split("-", 1)
    start_s, end_s = parse_timestamp(start_str), parse_timestamp(end_str)
    if end_s <= start_s:
        raise ValueError(f"Clip end must be after start: {clip!r}")
    return start_s, end_s


def clip_audio(src: Path, dest: Path, start_s: float, end_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_s), "-to", str(end_s), "-i", str(src),
        "-ar", "16000", "-ac", "1", str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def run_bench(
    cfg: Config,
    url: str,
    model_keys: list[str],
    clip: str | None,
    ref_path: str | None,
    console,
    language: str | None = None,
) -> None:
    console.print(f"Fetching audio for {url}...")
    bench_dir = cfg.paths.audio_dir / "bench"
    audio_path = fetch.download_audio(url, bench_dir, cfg)
    clip_duration: float | None = None

    if clip:
        start_s, end_s = parse_clip(clip)
        clipped_path = audio_path.with_name(audio_path.stem + "_clip.wav")
        clip_audio(audio_path, clipped_path, start_s, end_s)
        audio_path.unlink(missing_ok=True)
        audio_path = clipped_path
        clip_duration = end_s - start_s

    reference = Path(ref_path).read_text(encoding="utf-8").strip() if ref_path else None

    try:
        for raw_key in model_keys:
            key = raw_key.strip()
            _bench_one_model(cfg, key, audio_path, clip_duration, reference, console, language)
    finally:
        audio_path.unlink(missing_ok=True)


def _bench_one_model(
    cfg: Config,
    key: str,
    audio_path: Path,
    clip_duration: float | None,
    reference: str | None,
    console,
    language: str | None = None,
) -> None:
    model_cfg = cfg.asr.models.get(key)
    if model_cfg is None:
        console.print(f"[red]Unknown model key: {key}[/red]")
        return
    model_dir = cfg.paths.models_dir / key
    if not model_dir.exists():
        console.print(
            f"[yellow]Model '{key}' not converted yet. Run: "
            f"uv run --extra convert python scripts/convert_model.py {key}[/yellow]"
        )
        return

    transcriber = Transcriber(model_dir, model_cfg, cfg.asr)
    t0 = time.time()
    try:
        # A pinned model uses its own language; a multilingual one defaults to English.
        result = transcriber.transcribe(audio_path, language or model_cfg.language or "en")
    except ValueError as exc:
        console.print(f"[red]{key}: {escape(str(exc))}[/red]")
        return
    finally:
        elapsed = time.time() - t0
        transcriber.unload()

    text = " ".join(s.text for s in result.segments)
    duration = clip_duration or (result.segments[-1].end_s if result.segments else elapsed)
    rtf = elapsed / duration if duration else float("nan")

    console.print(f"\n[bold]{key}[/bold]  device={result.device} compute_type={result.compute_type}")
    console.print(f"  time: {elapsed:.1f}s for {duration:.1f}s audio  (RTF: {rtf:.2f})")
    st = result.stats
    console.print(
        f"  clips: {st.clips}  resplit (hit token limit): {st.resplit}  "
        f"still truncated: {st.truncated_unresolved}"
    )
    if reference is not None:
        import jiwer

        console.print(f"  WER: {jiwer.wer(reference, text):.3f}  CER: {jiwer.cer(reference, text):.3f}")
    console.print(f"  text: {text[:300]}")
