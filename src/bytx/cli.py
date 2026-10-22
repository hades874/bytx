"""bytx command-line interface."""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import compare as compare_module
from . import export, fetch, langs, models, pipeline, platforms, quality, rag, store
from .config import Config, load_config
from .cuda_dll import ensure_cuda_dll_paths
from .llm import OllamaClient, OllamaError
from .textio import configure_utf8_io

configure_utf8_io()

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

_config_option = typer.Option(
    None, "--config",
    help="Path to a config.toml with your overrides; its folder is the data folder "
    "(default: ~/bytx, or $BYTX_HOME)",
)


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


_OLLAMA_OPTIONAL = "It's optional: install it from https://ollama.com (or start it), then run `bytx setup`."


def _ollama_client(cfg: Config) -> OllamaClient:
    """A client for a running Ollama, or exit with how to get one."""
    client = OllamaClient(cfg.ollama)
    if not client.is_reachable():
        client.close()
        console.print(
            f"[yellow]This needs Ollama, which isn't running at {cfg.ollama.host}. "
            f"{_OLLAMA_OPTIONAL}[/yellow]"
        )
        raise typer.Exit(1)
    return client


_LANG_CHOICES = ", ".join(langs.SUPPORTED)
_PLATFORM_CHOICES = ", ".join(platforms.SUPPORTED)


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


def _parse_platform(value: str | None) -> str | None:
    """CLI --platform value -> platform name; None when the option was not given."""
    if value is None:
        return None
    name = value.strip().lower()
    if name not in platforms.SUPPORTED:
        console.print(f"[red]Unknown platform {value!r}. Choose from: {_PLATFORM_CHOICES}.[/red]")
        raise typer.Exit(1)
    return name


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

    # deno is the JS runtime yt-dlp needs for YouTube only; Facebook works without it.
    for tool, note in (("ffmpeg", ""), ("deno", " (needed for YouTube; Facebook works without it)")):
        path = shutil.which(tool)
        if path:
            console.print(f"[green]OK[/green]  {tool}: {path}")
        else:
            console.print(f"[red]MISSING[/red]  {tool} not found on PATH{note}")
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
        try:
            model_dir = models.local_path(cfg.asr.models[key])
        except models.ModelsMissing as exc:
            console.print(f"[yellow]MISSING[/yellow]  {escape(str(exc))}")
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

    # Ollama only powers search by meaning, Q&A and summaries, so it never fails the check.
    try:
        with OllamaClient(cfg.ollama) as client:
            if not client.is_reachable():
                console.print(
                    f"[yellow]WARN[/yellow]  Ollama not reachable at {cfg.ollama.host} (needed only for "
                    f"search by meaning, Q&A and summaries). {_OLLAMA_OPTIONAL}"
                )
            else:
                console.print(f"[green]OK[/green]  Ollama reachable at {cfg.ollama.host}")
                for model in (cfg.ollama.embed_model, cfg.ollama.chat_model):
                    if client.has_model(model):
                        console.print(f"[green]OK[/green]  Ollama model '{model}' present")
                    else:
                        console.print(
                            f"[yellow]WARN[/yellow]  Ollama model '{model}' missing. Run `bytx setup`."
                        )
    except OllamaError as exc:
        console.print(f"[yellow]WARN[/yellow]  Ollama check failed: {exc}")

    if not ok:
        raise typer.Exit(1)
    console.print("\n[bold green]All checks passed.[/bold green]")


@app.command()
def setup(config: str | None = _config_option, force: bool = False) -> None:
    """Download every configured ASR model and pull the Ollama models."""
    cfg = _load(config)

    try:
        cfg.asr.active_model()
        model_keys = cfg.asr.required_model_keys()
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    for key in model_keys:
        console.print(f"[bold]Setting up ASR model[/bold]: {key}")
        try:
            path = models.download(cfg.asr.models[key], force=force)
        except Exception as exc:  # noqa: BLE001 - config, network and Hugging Face errors
            console.print(f"[red]Model setup failed:[/red] {escape(str(exc))}")
            raise typer.Exit(1) from exc
        console.print(f"[green]OK[/green]  {key}: {path}")

    with OllamaClient(cfg.ollama) as client:
        ollama_running = client.is_reachable()
    if not ollama_running:
        console.print(
            f"[yellow]Skipped the Q&A and summary models: Ollama isn't running at {cfg.ollama.host}. "
            f"{_OLLAMA_OPTIONAL}[/yellow]"
        )
    else:
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
    try:
        result = pipeline.run(conn, cfg, limit=limit, summarize_enabled=summarize_enabled)
    except models.ModelsMissing as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    detected = f"language-detected={result.detected} " if result.detected else ""
    console.print(
        f"[bold]Done.[/bold] {detected}transcribed={result.transcribed} indexed={result.indexed} "
        f"summarized={result.summarized} failed={result.failed}"
    )
    if result.stopped:
        console.print(f"[yellow]{escape(result.stopped)}[/yellow]")
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
    platform = _parse_platform(platform)
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
            with _ollama_client(cfg) as client:
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
        with _ollama_client(cfg) as client:
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
        with _ollama_client(cfg) as client:
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
def remove(
    video_id: str,
    yes: bool = typer.Option(False, "--yes", help="Delete (default: only list what would be deleted)"),
    config: str | None = _config_option,
) -> None:
    """Remove a video completely, so it can be added again (e.g. with the right --lang)."""
    cfg = _load(config)
    conn = _conn(cfg)
    try:
        removal = pipeline.remove_video(conn, cfg, video_id)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    if removal.empty:
        console.print(
            f"[red]No such video: {escape(video_id)}[/red] (not in the library, its folders or the ledger)"
        )
        raise typer.Exit(1)

    v = removal.video
    lang = escape(f"[{v.language}]") if v and v.language else ""
    console.print(f"video {video_id}: {escape(v.title or v.url) if v else '(not in the database)'}  {lang}")
    console.print(f"  database row:     {'yes' if v else 'no'}")
    console.print(f"  data folder:      {removal.data_dir or 'none'}")
    console.print(f"  transcript copy:  {removal.shared_folder or 'none'}")
    console.print(f"  ledger entry:     {'yes' if removal.ledger_entry else 'no'}")
    if not yes:
        console.print("Dry run: nothing was deleted. Run again with --yes to delete the above.")
        return
    pipeline.remove_video(conn, cfg, video_id, apply=True)
    console.print(f"[green]Removed {video_id}.[/green] Re-add it with `bytx add <url> --lang bn|en`.")


@app.command(name="quality")
def quality_cmd(
    video_id: str | None = typer.Argument(None, help="One video to inspect in detail"),
    flagged: bool = typer.Option(False, "--flagged", help="Only list transcripts with quality flags"),
    config: str | None = _config_option,
) -> None:
    """Show how far each transcript can be trusted (words/min, gaps, low-confidence clips)."""
    cfg = _load(config)
    conn = _conn(cfg)
    if video_id is not None:
        video = store.get_video(conn, video_id)
        if video is None:
            console.print(f"[red]No such video: {video_id}[/red]")
            raise typer.Exit(1)
        videos = [video]
    else:
        videos = [v for v in store.list_videos(conn) if v.status not in ("queued", "failed")]

    rows = []
    for v in videos:
        segments = store.get_segments(conn, v.id)
        if segments:
            rows.append((v, quality.assess(v, segments, cfg.quality)))
    rows.sort(key=lambda r: (not r[1].suspect, r[0].created_at or ""))  # flagged first
    if flagged:
        rows = [r for r in rows if r[1].suspect]
    if not rows:
        console.print("No transcripts have quality flags." if flagged else "No transcripts to assess.")
        return

    if video_id is not None:
        _print_quality_detail(*rows[0])
        return

    table = Table(title="Transcript quality")
    table.add_column("ID", no_wrap=True)
    table.add_column("Title")
    table.add_column("Words/min", justify="right", no_wrap=True)
    table.add_column("Covered", justify="right", no_wrap=True)
    table.add_column("Gaps", justify="right", no_wrap=True)
    table.add_column("Low-conf", justify="right", no_wrap=True)
    table.add_column("Flags")
    for v, q in rows:
        table.add_row(
            v.id, escape((v.title or "")[:40]),
            f"{q.words_per_minute:.0f}" if q.words_per_minute is not None else "",
            f"{q.coverage:.0%}" if q.coverage is not None else "",
            str(len(q.gaps)),
            str(len(q.low_confidence)) if q.confidence_known else "-",
            escape("; ".join(q.flags)),
        )
    console.print(table)
    console.print(f"{sum(1 for _v, q in rows if q.suspect)} of {len(rows)} transcript(s) flagged.")


def _print_quality_detail(video: store.Video, q: quality.Quality) -> None:
    console.print(f"[bold]{escape(video.title or video.id)}[/bold]  ({video.url})")
    wpm = f"{q.words_per_minute:.0f}" if q.words_per_minute is not None else "?"
    cov = f"{q.coverage:.0%}" if q.coverage is not None else "?"
    console.print(f"{q.words} words, {wpm} words/min, {cov} of the video covered, {q.segments} clips")
    if q.confidence_known:
        console.print(
            f"mean clip log-probability {q.mean_avg_logprob:.2f}; {len(q.low_confidence)} low-confidence"
        )
    else:
        console.print(
            "[dim]No per-clip confidence recorded (transcribed before it was tracked, "
            "or from subtitles).[/dim]"
        )
    for g in q.gaps:
        span = f"{quality.format_time(g.start_s)}-{quality.format_time(g.end_s)}"
        console.print(f"  gap {span} ({g.seconds:.0f}s)")
    for idx in q.low_confidence:
        console.print(f"  low-confidence clip #{idx}")
    for flag in q.flags:
        console.print(f"[yellow]! {escape(flag)}[/yellow]")
    if not q.flags:
        console.print("[green]No quality flags.[/green]")


@app.command()
def reexport(
    video_ids: list[str] = typer.Argument(None, help="Videos to rewrite"),
    all_: bool = typer.Option(False, "--all", help="Rewrite every transcribed video"),
    config: str | None = _config_option,
) -> None:
    """Rebuild transcript folders and INDEX.md from the database (no speech recognition).

    Rewrites data/videos/<id>/ and yt-transcribed-files/ in the current layout, with
    metadata, quality flags and the glossary-cleaned copy. Run it after editing
    glossary.txt or the [quality] thresholds.
    """
    if not video_ids and not all_:
        console.print("[red]Give one or more video ids, or --all.[/red]")
        raise typer.Exit(1)
    cfg = _load(config)
    conn = _conn(cfg)
    done = pipeline.reexport(conn, cfg, None if all_ else list(video_ids))
    console.print(f"Rewrote {len(done)} transcript(s) under {cfg.paths.transcribed_files_dir}.")
    skipped = [i for i in (video_ids or []) if i not in done]
    if skipped:
        console.print(f"[yellow]Skipped (unknown or not transcribed): {', '.join(skipped)}[/yellow]")


def _load_reference(
    video: store.Video, cfg: Config, against: str | None
) -> tuple[list[store.Segment], str]:
    """The second transcript to compare with, and a name for it."""
    if against:
        path = Path(against)
        if not path.is_file():
            console.print(f"[red]No such file: {against}[/red]")
            raise typer.Exit(1)
        text = path.read_text(encoding="utf-8")
        parse = fetch.parse_json3_captions if path.suffix.lower() == ".json3" else fetch.parse_vtt
        return parse(text), path.name

    if video.platform != platforms.YOUTUBE:
        console.print("[red]Only YouTube videos have auto-captions to compare with. Use --against.[/red]")
        raise typer.Exit(1)
    language = video.language or cfg.asr.default_language
    try:
        info = fetch.get_video_info(video.url, cfg)
        caption_url = info.auto_subtitles.get(language)
        if caption_url is not None:
            return fetch.fetch_auto_caption_segments(caption_url), "YouTube auto-captions"
    except Exception as exc:  # noqa: BLE001 - yt-dlp and httpx raise many kinds of errors
        console.print(f"[red]Could not fetch auto-captions: {escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    heard = info.auto_spoken_language
    reason = (
        f"YouTube hears this video as {heard!r}, not {langs.display_name(language)}, so its "
        "auto-captions are in another language"
        if heard
        else f"YouTube has no {langs.display_name(language)} auto-captions for this video"
    )
    console.print(f"[red]{reason}. Use --against with a caption file.[/red]")
    raise typer.Exit(1)


@app.command(name="compare")
def compare_cmd(
    video_id: str,
    against: str | None = typer.Option(
        None, "--against",
        help="Compare with this caption file (WebVTT, or YouTube .json3) instead of "
        "fetching YouTube's auto-captions",
    ),
    threshold: float = typer.Option(
        compare_module.DEFAULT_THRESHOLD, "--threshold", min=0.0, max=1.0,
        help="Report segments less similar than this (0-1)",
    ),
    output: str | None = typer.Option(
        None, "--output", help="Write the report here (default: compare.md in the transcript folder)"
    ),
    config: str | None = _config_option,
) -> None:
    """Cross-check a transcript against an independent one to find likely errors.

    Compares each segment with YouTube's auto-generated captions for the same
    moment. Where the two disagree is where a name, number or sentence is most
    likely wrong; it is a spot to listen to, not proof. YouTube's Bangla captions
    are weak, so this is most useful for English and mixed-language videos.
    """
    cfg = _load(config)
    conn = _conn(cfg)
    video = store.get_video(conn, video_id)
    segments = store.get_segments(conn, video_id) if video else []
    if video is None or not segments:
        console.print(f"[red]No transcript for video: {video_id}[/red]")
        raise typer.Exit(1)

    reference, reference_name = _load_reference(video, cfg, against)
    result = compare_module.compare(segments, reference, threshold)
    report = compare_module.render_markdown(video, result, reference_name, threshold)
    if output:
        target: Path | None = Path(output)
    else:
        folder = export.find_transcript_folder(cfg.paths.transcribed_files_dir, video_id)
        target = folder / "compare.md" if folder else None
    if target is not None:
        target.write_text(report, encoding="utf-8")

    if result.mean_similarity is None:
        console.print(
            "[yellow]The captions have no text inside any transcript segment; nothing to compare.[/yellow]"
        )
    else:
        console.print(
            f"Mean similarity {result.mean_similarity:.0%} over {result.compared} segments; "
            f"{len(result.disagreements)} below {threshold:.0%}; "
            f"{len(result.missed)} stretch(es) of captioned speech missing from the transcript."
        )
    for d in result.disagreements[:5]:
        console.print(f"[dim]{quality.format_time(d.start_s)}[/dim] {d.similarity:.0%}")
        console.print(f"  transcript: {escape(d.text)}")
        console.print(f"  captions:   {escape(d.ref_text)}")
    if len(result.disagreements) > 5:
        console.print(f"[dim]... and {len(result.disagreements) - 5} more[/dim]")
    if target is not None:
        console.print(f"Full report: {target}")
    else:
        console.print("[dim]Pass --output to save the full report.[/dim]")


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
