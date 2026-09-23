"""Staged, resumable pipeline runner.

Each video moves through statuses: queued -> transcribed -> indexed ->
summarized (or failed at any point). Running the pipeline again after a
crash or Ctrl+C simply re-queries each stage's input status, so already
finished videos are skipped and nothing is redone.

Stage 1 (transcribe) uses the whole GPU for faster-whisper; the model is
unloaded before stage 2/3 touch Ollama, so the two never compete for the
4GB of VRAM at the same time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from . import export, fetch, rag, store, summarize
from .config import Config
from .llm import OllamaClient
from .transcribe import Transcriber

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    transcribed: int = 0
    indexed: int = 0
    summarized: int = 0
    failed: int = 0


def add_targets(conn, cfg: Config, targets: list[str]) -> list[str]:
    """Expands URLs/playlists/channels/text-file targets into individual
    video URLs and inserts new queued rows. Already-known video ids are
    skipped. Returns the list of newly queued video ids.
    """
    new_ids: list[str] = []
    for target in targets:
        for url in fetch.expand_to_video_urls(target, cfg):
            vid = fetch.extract_video_id(url) or url
            if store.video_exists(conn, vid):
                continue
            store.upsert_video(conn, store.Video(id=vid, url=url, status="queued"))
            new_ids.append(vid)
    return new_ids


def transcribe_one(conn, cfg: Config, transcriber: Transcriber, video: store.Video) -> None:
    """Runs stage 1 for a single video: fetch metadata, prefer human
    subtitles over ASR, store segments, write exports, delete audio.
    """
    info = fetch.get_video_info(video.url, cfg)
    store.upsert_video(
        conn,
        store.Video(
            id=info.id,
            url=info.url,
            title=info.title,
            channel=info.channel,
            channel_id=info.channel_id,
            upload_date=info.upload_date,
            duration=info.duration,
            description=info.description,
        ),
    )

    use_human_subs = info.human_subtitle_url is not None and not cfg.fetch.force_asr
    if use_human_subs:
        segments = fetch.fetch_human_subtitle_segments(info.human_subtitle_url)
        source, asr_model = "human_subs", None
    else:
        audio_path = fetch.download_audio(info.url, cfg.paths.audio_dir, cfg)
        try:
            result = transcriber.transcribe(audio_path)
            segments = result.segments
        finally:
            audio_path.unlink(missing_ok=True)
        source, asr_model = "asr", cfg.asr.model

    store.replace_segments(conn, info.id, segments)
    store.set_video_source(conn, info.id, source, asr_model)
    updated = store.get_video(conn, info.id)
    assert updated is not None
    export.write_info_json(cfg.paths.videos_dir, updated)
    export.write_all(cfg.paths.videos_dir, updated, segments)
    store.set_video_status(conn, info.id, "transcribed")


def index_one(conn, cfg: Config, client: OllamaClient, video: store.Video) -> None:
    segments = store.get_segments(conn, video.id)
    rag.index_video(conn, client, cfg, video.id, segments)
    store.set_video_status(conn, video.id, "indexed")


def summarize_one(conn, cfg: Config, client: OllamaClient, video: store.Video) -> None:
    segments = store.get_segments(conn, video.id)
    summary = summarize.summarize_video(client, cfg, video, segments)
    store.upsert_summary(conn, video.id, cfg.ollama.chat_model, summary)
    export.write_summary_md(cfg.paths.videos_dir, video, summary)
    store.set_video_status(conn, video.id, "summarized")


def _run_stage(
    conn, videos: list[store.Video], stage_name: str, fn, result: RunSummary, counter_attr: str
) -> None:
    for video in videos:
        try:
            fn(video)
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the queue
            logger.exception("%s failed for %s", stage_name, video.id)
            store.set_video_status(conn, video.id, "failed", error=str(exc), increment_attempts=True)
            result.failed += 1
        else:
            setattr(result, counter_attr, getattr(result, counter_attr) + 1)


def run(conn, cfg: Config, limit: int | None = None, summarize_enabled: bool | None = None) -> RunSummary:
    result = RunSummary()
    should_summarize = cfg.summarize.enabled if summarize_enabled is None else summarize_enabled

    queued = store.list_videos(conn, status="queued")
    if limit is not None:
        queued = queued[:limit]
    if queued:
        model_cfg = cfg.asr.active_model()
        model_dir = cfg.paths.models_dir / cfg.asr.model
        transcriber = Transcriber(model_dir, model_cfg, cfg.asr)
        _run_stage(
            conn, queued, "transcribe",
            lambda v: transcribe_one(conn, cfg, transcriber, v),
            result, "transcribed",
        )
        transcriber.unload()

    to_index = store.list_videos(conn, status="transcribed")
    if to_index:
        with OllamaClient(cfg.ollama) as client:
            _run_stage(
                conn, to_index, "index",
                lambda v: index_one(conn, cfg, client, v),
                result, "indexed",
            )

    if should_summarize:
        to_summarize = store.list_videos(conn, status="indexed")
        if to_summarize:
            with OllamaClient(cfg.ollama) as client:
                _run_stage(
                    conn, to_summarize, "summarize",
                    lambda v: summarize_one(conn, cfg, client, v),
                    result, "summarized",
                )

    return result


def retry_failed(conn) -> int:
    failed = store.list_videos(conn, status="failed")
    for v in failed:
        store.set_video_status(conn, v.id, "queued", error=None)
    return len(failed)
