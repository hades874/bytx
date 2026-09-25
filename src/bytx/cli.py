"""bytx command-line interface."""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import export, langs, pipeline, platforms, rag, store
from .config import Config, load_config
from .cuda_dll import ensure_cuda_dll_paths
from .llm import OllamaClient, OllamaError
from .textio import configure_utf8_io

configure_utf8_io()

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

_config_option = typer.Option(None, "--config", help="Path to config.toml (default: project root)")


def _load(config_path: str | None) -> Config:
    try:
        cfg = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    cfg.ensure_dirs()
    return cfg


def _conn(cfg: Config):
    return store.connect(cfg.paths.db_path)


_LANG_CHOICES = ", ".join(langs.SUPPORTED)
_PLATFORMS = (platforms.YOUTUBE, platforms.FACEBOOK)
_PLATFORM_CHOICES = ", ".join(_PLATFORMS)


def _parse_lang(value: str | None, allow_auto: bool = False) -> str | None:
    """CLI --lang value -> language code; None when the option was not given.
    `allow_auto` also accepts "auto" (pipeline.AUTO: detect at run time)."""
    if value is None:
        return None
    if allow_auto and value.strip().lower() == pipeline.AUTO:
        return pipeline.AUTO
    code = langs.normalize(value)
    if code is None:
        choices = f"{pipeline.AUTO}, {_LANG_CHOICES}" if allow_auto else _LANG_CHOICES
        console.print(f"[red]Unsupported language {value!r}. Use one of: {choices}.[/red]")
        raise typer.Exit(1)
    return code


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@app.command()
def doctor(config: str | None = _config_option) -> None:
    """Check that ffmpeg, yt-dlp, the ASR model, and Ollama are ready."""
    cfg = _load(config)
    ok = True

    import shutil

    for tool in ("ffmpeg", "deno"):
        path = shutil.which(tool)
        if path:
            console.print(f"[green]OK[/green]  {tool}: {path}")
        else:
            console.print(f"[red]MISSING[/red]  {tool} not found on PATH")
            ok = False

    try:
        proc = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=15)
        console.print(f"[green]OK[/green]  yt-dlp: {proc.stdout.strip()}")
    except Exception as exc:
        console.print(f"[red]MISSING[/red]  yt-dlp: {exc}")
        ok = False

    from .transcribe import ModelLoadError, Transcriber

    try:
        cfg.asr.active_model()
        model_keys = cfg.asr.required_model_keys()
    except ValueError as exc:
        console.print(f"[red]FAIL[/red]  ASR config: {escape(str(exc))}")
        raise typer.Exit(1) from exc

    for key in model_keys:
        model_dir = cfg.paths.models_dir / key
        if not model_dir.exists():
            console.print(
                f"[yellow]MISSING[/yellow]  ASR model '{key}' not found at {model_dir}. "
                f"Run `bytx setup`."
            )
            ok = False
            continue
        console.print(f"[green]OK[/green]  ASR model '{key}' present at {model_dir}")
        try:
            t = Transcriber(model_dir, cfg.asr.models[key], cfg.asr)
            device, compute_type = t.device, t.compute_type
            console.print(
                f"[green]OK[/green]  ASR '{key}' loads on device={device} compute_type={compute_type}"
            )
            if device != "cuda" or compute_type != cfg.asr.compute_type:
                console.print(
                    f"  [yellow]note:[/yellow] fell back from cuda/{cfg.asr.compute_type} "
                    f"(see logs with -v for why)"
                )
            t.unload()
        except ModelLoadError as exc:
            console.print(f"[red]FAIL[/red]  ASR model '{key}' failed to load: {escape(str(exc))}")
            ok = False

    try:
        with OllamaClient(cfg.ollama) as client:
            if not client.is_reachable():
                console.print(f"[red]FAIL[/red]  Ollama not reachable at {cfg.ollama.host}")
                ok = False
            else:
                console.print(f"[green]OK[/green]  Ollama reachable at {cfg.ollama.host}")
                for model in (cfg.ollama.embed_model, cfg.ollama.chat_model):
                    if client.has_model(model):
                        console.print(f"[green]OK[/green]  Ollama model '{model}' present")
                    else:
                        console.print(f"[yellow]MISSING[/yellow]  Ollama model '{model}'. Run `bytx setup`.")
                        ok = False
    except OllamaError as exc:
        console.print(f"[red]FAIL[/red]  Ollama check failed: {exc}")
        ok = False

    if not ok:
        raise typer.Exit(1)
    console.print("\n[bold green]All checks passed.[/bold green]")


@app.command()
def setup(config: str | None = _config_option, force: bool = False) -> None:
    """Convert/download every configured ASR model and pull Ollama models."""
    cfg = _load(config)
    scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    convert_script = scripts_dir / "convert_model.py"

    try:
        cfg.asr.active_model()
        model_keys = cfg.asr.required_model_keys()
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    for key in model_keys:
        console.print(f"[bold]Setting up ASR model[/bold]: {key}")
        cmd = [sys.executable, str(convert_script), key]
        if force:
            cmd.append("--force")
        if config:
            cmd.extend(["--config", config])
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            console.print("[red]Model setup failed.[/red] If this is the first run, install the 'convert' "
                           "extra: uv sync --extra convert")
            raise typer.Exit(1)

    for model in (cfg.ollama.embed_model, cfg.ollama.chat_model):
        console.print(f"[bold]Pulling Ollama model[/bold]: {model}")
        proc = subprocess.run(["ollama", "pull", model])
        if proc.returncode != 0:
            console.print(f"[red]Failed to pull {model}[/red]")
            raise typer.Exit(1)

    console.print("[bold green]Setup complete.[/bold green] Run `bytx doctor` to verify.")


@app.command()
def add(
    targets: list[str] = typer.Argument(
        ..., help="YouTube or Facebook video/playlist/channel URLs, or a path to a text file of URLs"
    ),
    run: bool = typer.Option(False, "--run", help="Process the queue immediately after adding"),
    lang: str | None = typer.Option(
        None,
        "--lang",
        help=(
            f"Language of these videos ({_LANG_CHOICES}); default: asr.default_language in config.toml. "
            "'auto' decides per video when it is processed (Bangla vs English, from a transcribed sample)"
        ),
    ),
    force_asr: bool = typer.Option(
        False, "--force-asr",
        help="Ignore uploader subtitles, always use ASR (only takes effect together with --run)",
    ),
    limit: int | None = typer.Option(None, help="With --run, process at most this many videos"),
    config: str | None = _config_option,
) -> None:
    """Queue one or more videos/playlists/channels for transcription."""
    language = _parse_lang(lang, allow_auto=True)
    cfg = _load(config)
    if force_asr:
        cfg = dataclasses.replace(cfg, fetch=dataclasses.replace(cfg.fetch, force_asr=True))
    conn = _conn(cfg)
    console.print("Expanding targets...")
    new_ids = pipeline.add_targets(conn, cfg, targets, language=language)
    console.print(f"[green]Queued {len(new_ids)} new video(s).[/green]")
    if run:
        _do_run(conn, cfg, limit)


@app.command()
def run(
    limit: int | None = typer.Option(None, help="Process at most this many videos this run"),
    no_summarize: bool = typer.Option(False, "--no-summarize", help="Skip the summarization stage"),
    config: str | None = _config_option,
) -> None:
    """Process the queue: transcribe, index, and (by default) summarize."""
    cfg = _load(config)
    conn = _conn(cfg)
    _do_run(conn, cfg, limit, summarize_enabled=(not no_summarize))


def _do_run(conn, cfg: Config, limit: int | None, summarize_enabled: bool | None = None) -> None:
    ensure_cuda_dll_paths()
    result = pipeline.run(conn, cfg, limit=limit, summarize_enabled=summarize_enabled)
    detected = f"language-detected={result.detected} " if result.detected else ""
    console.print(
        f"[bold]Done.[/bold] {detected}transcribed={result.transcribed} indexed={result.indexed} "
        f"summarized={result.summarized} failed={result.failed}"
    )
    if result.failed:
        console.print(
            "[yellow]Some videos failed. Run `bytx status` for details, "
            "`bytx retry` to requeue.[/yellow]"
        )


@app.command()
def status(config: str | None = _config_option) -> None:
    """Show a summary of videos by status."""
    cfg = _load(config)
    conn = _conn(cfg)
    videos = store.list_videos(conn)
    if not videos:
        console.print("No videos in the library yet. Use `bytx add <url>`.")
        return

    counts: dict[str, int] = {}
    for v in videos:
        counts[v.status] = counts.get(v.status, 0) + 1

    table = Table(title="Status summary")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for s, c in sorted(counts.items()):
        table.add_row(s, str(c))
    console.print(table)

    failed = [v for v in videos if v.status == "failed"]
    if failed:
        console.print("\n[bold red]Failed videos:[/bold red]")
        for v in failed:
            console.print(f"  {v.id}  {v.title or v.url}  [red]{v.error}[/red]")


@app.command(name="list")
def list_videos_cmd(
    channel: str | None = typer.Option(None, help="Filter by channel name"),
    status_filter: str | None = typer.Option(None, "--status", help="Filter by status"),
    lang: str | None = typer.Option(None, "--lang", help=f"Only videos in this language ({_LANG_CHOICES})"),
    platform: str | None = typer.Option(
        None, "--platform", help=f"Only videos from this platform ({_PLATFORM_CHOICES})"
    ),
    config: str | None = _config_option,
) -> None:
    """List videos in the library."""
    language = _parse_lang(lang)
    if platform is not None and platform not in _PLATFORMS:
        console.print(f"[red]Unknown platform {platform!r}. Choose from: {_PLATFORM_CHOICES}.[/red]")
        raise typer.Exit(1)
    cfg = _load(config)
    conn = _conn(cfg)
    videos = store.list_videos(
        conn, status=status_filter, channel_id=None, language=language, platform=platform
    )
    if channel:
        videos = [v for v in videos if v.channel == channel]

    table = Table()
    table.add_column("ID", no_wrap=True)  # ids are copied into `bytx show`; never truncate them
    table.add_column("Title")
    table.add_column("Channel")
    table.add_column("Lang")
    table.add_column("Platform", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", justify="right", no_wrap=True)
    for v in videos:
        table.add_row(
            v.id, (v.title or "")[:60], v.channel or "", v.language or "", v.platform, v.status,
            _fmt_hms(v.duration) if v.duration else "",
        )
    console.print(table)


@app.command()
def show(video_id: str, config: str | None = _config_option) -> None:
    """Print a video's transcript."""
    cfg = _load(config)
    conn = _conn(cfg)
    video = store.get_video(conn, video_id)
    if video is None:
        console.print(f"[red]No such video: {video_id}[/red]")
        raise typer.Exit(1)
    segments = store.get_segments(conn, video_id)
    lang_label = f"  {escape(f'[{video.language}]')}" if video.language else ""
    console.print(f"[bold]{video.title or video.id}[/bold]{lang_label}  ({video.url})")
    for s in segments:
        console.print(f"[dim]{_fmt_hms(s.start_s)}[/dim]  {s.text}")


@app.command(name="export")
def export_cmd(
    video_id: str,
    fmt: str = typer.Option("txt", "--format", help="txt, srt, vtt, or json"),
    output: str | None = typer.Option(None, "--output", help="Write to this path instead of stdout"),
    config: str | None = _config_option,
) -> None:
    """Export a video's transcript in the given format."""
    cfg = _load(config)
    conn = _conn(cfg)
    video = store.get_video(conn, video_id)
    if video is None:
        console.print(f"[red]No such video: {video_id}[/red]")
        raise typer.Exit(1)
    segments = store.get_segments(conn, video_id)
    summary = store.get_summary(conn, video_id)
    try:
        rendered = export.render(fmt, video, segments, summary=summary)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if output:
        Path(output).write_text(rendered, encoding="utf-8")
        console.print(f"Wrote {output}")
    else:
        print(rendered)


@app.command()
def search(
    query: str,
    semantic: bool = typer.Option(False, "--semantic", help="Search by meaning instead of keyword"),
    channel: str | None = typer.Option(None, help="Filter by channel name"),
    limit: int = typer.Option(20, help="Max results"),
    config: str | None = _config_option,
) -> None:
    """Search transcripts by keyword (default) or meaning (--semantic)."""
    cfg = _load(config)
    conn = _conn(cfg)

    if semantic:
        try:
            with OllamaClient(cfg.ollama) as client:
                hits = rag.semantic_search(conn, client, cfg, query, top_k=limit, channel=channel)
        except OllamaError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        for hit in hits:
            link = platforms.timestamp_url(hit.video_id, hit.start_s)
            title = hit.title or hit.video_id
            console.print(f"[bold]{title}[/bold]  {_fmt_hms(hit.start_s)}  ({hit.score:.2f})")
            console.print(f"  {escape(hit.text[:200])}")
            console.print(f"  [dim]{link}[/dim]\n")
    else:
        try:
            hits = store.search_fts(conn, query, limit=limit, channel=channel)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        for hit in hits:
            link = platforms.timestamp_url(hit.video_id, hit.start_s)
            console.print(f"[bold]{hit.title or hit.video_id}[/bold]  {_fmt_hms(hit.start_s)}")
            console.print(f"  {escape(hit.snippet)}")  # FTS marks the match as [word]
            console.print(f"  [dim]{link}[/dim]\n")


@app.command()
def ask(
    question: str,
    video: str | None = typer.Option(None, "--video", help="Restrict to a single video id"),
    config: str | None = _config_option,
) -> None:
    """Ask a question answered from your transcript library via the local LLM."""
    cfg = _load(config)
    conn = _conn(cfg)
    try:
        with OllamaClient(cfg.ollama) as client:
            answer, hits = rag.ask(conn, client, cfg, question, video_id=video)
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(answer)
    if hits:
        console.print("\n[dim]Sources:[/dim]")
        for hit in hits:
            console.print(f"  [dim]{rag.format_citation(hit)}[/dim]")


@app.command()
def summarize(
    video_id: str | None = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Summarize every indexed/summarized video"),
    config: str | None = _config_option,
) -> None:
    """Generate or regenerate a summary for one video or all videos."""
    if not video_id and not all_:
        console.print("[red]Provide a video id or --all.[/red]")
        raise typer.Exit(1)
    cfg = _load(config)
    conn = _conn(cfg)

    if all_:
        targets = [v for v in store.list_videos(conn) if v.status in ("indexed", "summarized")]
    else:
        v = store.get_video(conn, video_id)
        if v is None:
            console.print(f"[red]No such video: {video_id}[/red]")
            raise typer.Exit(1)
        targets = [v]

    try:
        with OllamaClient(cfg.ollama) as client:
            for v in targets:
                pipeline.summarize_one(conn, cfg, client, v)
                console.print(f"[green]Summarized[/green] {v.id}")
    except OllamaError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def retry(config: str | None = _config_option) -> None:
    """Requeue failed videos so the next `bytx run` retries them."""
    cfg = _load(config)
    conn = _conn(cfg)
    count = pipeline.retry_failed(conn)
    console.print(f"Requeued {count} failed video(s).")


@app.command()
def bench(
    url: str,
    models: str | None = typer.Option(
        None, help="Comma-separated model keys to compare (default: config's model)"
    ),
    clip: str | None = typer.Option(None, help="mm:ss-mm:ss range to clip before transcribing"),
    ref: str | None = typer.Option(None, help="Reference transcript file for WER/CER scoring"),
    lang: str | None = typer.Option(
        None, "--lang",
        help=f"Language to transcribe in ({_LANG_CHOICES}); default: each model's own language, else en",
    ),
    config: str | None = _config_option,
) -> None:
    """Compare ASR models on a clip from a real video."""
    from . import bench as bench_module

    language = _parse_lang(lang)
    cfg = _load(config)
    model_keys = models.split(",") if models else [cfg.asr.model]
    bench_module.run_bench(
        cfg, url, model_keys, clip=clip, ref_path=ref, console=console, language=language
    )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
