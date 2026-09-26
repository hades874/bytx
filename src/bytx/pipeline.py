"""Staged, resumable pipeline runner.

Each video moves through statuses: queued -> transcribed -> indexed ->
summarized (or failed at any point). Running the pipeline again after a
crash or Ctrl+C simply re-queries each stage's input status, so already
finished videos are skipped and nothing is redone.

Stage 1 (transcribe) uses the whole GPU for faster-whisper; the model is
unloaded before stage 2/3 touch Ollama, so the two never compete for the
4GB of VRAM at the same time.

Each video has a language (bn or en) that picks its ASR model and which
uploader subtitles are used. It is set with `bytx add --lang` (default:
`asr.default_language`) and stored on the video row. Whisper's language ID and
YouTube's metadata both mislabel Bangla speech that is full of English words
as English, so neither is used. `--lang auto` instead queues the video with no
language and the run decides it by transcribing a sample with the Bangla model
(see langdetect.py). Nothing ever auto-detects per chunk, so the language of
a transcript is always a deliberate, recorded choice.

After transcription every video also gets a quality assessment (quality.py) and a
glossary-cleaned copy of its text (glossary.py); the raw ASR text stays the source
of truth. `reexport` rebuilds all of those files from the database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import export, fetch, langdetect, langs, ledger, platforms, quality, rag, store, summarize
from . import glossary as glossary_mod
from .config import Config
from .llm import OllamaClient
from .transcribe import TranscriberPool, TranscribeStats

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    detected: int = 0
    transcribed: int = 0
    indexed: int = 0
    summarized: int = 0
    failed: int = 0


AUTO = "auto"  # add_targets(language=AUTO): decide each video's language at run time


def _identify(url: str, cfg: Config) -> store.Video | None:
    """The queued row for the video at `url`, or None (after logging why) when
    it is not a YouTube or Facebook video we can identify."""
    ref = platforms.parse_video_ref(url)
    if ref is not None:
        return store.Video(id=ref.id, url=url, platform=ref.platform)
    if platforms.platform_of_url(url) is None:
        logger.warning("Skipping %s: only YouTube and Facebook videos are supported", url)
        return None
    # A link with no id of its own (fb.watch/xyz, /share/v/xyz, an odd YouTube URL):
    # fetch it to learn which video it points to.
    try:
        info = fetch.get_video_info(url, cfg)
    except Exception as exc:  # noqa: BLE001 - yt-dlp raises many kinds of errors
        logger.warning(
            "Skipping %s: could not tell which video it points to (%s). If it needs a login, set "
            "cookies_from_browser under [fetch] in config.toml; otherwise add the full video URL.",
            url, exc,
        )
        return None
    return store.Video(id=info.id, url=info.url, platform=info.platform)


def add_targets(
    conn, cfg: Config, targets: list[str], language: str | None = None
) -> list[str]:
    """Expands URLs/playlists/channels/text-file targets into individual
    video URLs and inserts new queued rows in `language` (default:
    `asr.default_language`; `AUTO` leaves it unset for `run` to detect).
    Already-known video ids are skipped. Returns the list of newly queued
    video ids.
    """
    if language == AUTO:
        language = None
    else:
        language = language or cfg.asr.default_language
    seen_ids = ledger.load_seen_ids(ledger.ledger_path(cfg))
    new_ids: list[str] = []
    for target in targets:
        for url in fetch.expand_to_video_urls(target, cfg):
            video = _identify(url, cfg)
            if video is None or store.video_exists(conn, video.id) or video.id in seen_ids:
                continue
            video.language = language
            store.upsert_video(conn, video)
            new_ids.append(video.id)
    return new_ids


# Below this fraction of the video's duration detected as speech, VAD has
# probably dropped real speech (or the video is mostly music/silence).
_MIN_SPEECH_COVERAGE = 0.75


def _warn_on_suspect_asr(info: fetch.VideoInfo, stats: TranscribeStats) -> None:
    """Logs (never fails) when ASR output looks incomplete."""
    if stats.truncated_unresolved:
        logger.warning(
            "%s: %d clip(s) still hit Whisper's token limit after splitting; "
            "their text is probably cut off (try a lower asr.max_clip_seconds)",
            info.id, stats.truncated_unresolved,
        )
    if info.duration and stats.speech_seconds / info.duration < _MIN_SPEECH_COVERAGE:
        logger.warning(
            "%s: only %.0f%% of the video (%.0fs of %.0fs) was detected as speech; "
            "some speech may have been skipped (try a lower asr.vad_threshold)",
            info.id, 100 * stats.speech_seconds / info.duration,
            stats.speech_seconds, info.duration,
        )


def write_transcript_files(
    cfg: Config, video: store.Video, segments: list[store.Segment], summary: str | None = None
) -> quality.Quality:
    """Writes data/videos/<id>/ (info + every transcript format) for a
    transcribed video and returns its quality assessment."""
    assessment = quality.assess(video, segments, cfg.quality)
    export.write_info_json(cfg.paths.videos_dir, video)
    export.write_all(cfg.paths.videos_dir, video, segments, summary=summary, quality=assessment)
    return assessment


def write_shared_copy(
    cfg: Config, video: store.Video, segments: list[store.Segment], assessment: quality.Quality
) -> None:
    """Writes the human-facing copy under transcribed_files_dir (with a glossary-cleaned
    transcript alongside the raw one) and refreshes its INDEX.md."""
    cleaned, replacements = glossary_mod.load(cfg.asr.glossary).apply_segments(segments)
    export.write_transcript_copy(
        cfg.paths.transcribed_files_dir, video, segments, quality=assessment,
        cleaned=cleaned, glossary_replacements=replacements,
    )
    export.write_index(cfg.paths.transcribed_files_dir)


def detect_one(conn, cfg: Config, pool: TranscriberPool, video: store.Video) -> Path:
    """Decides the language of a video queued with `--lang auto` and stores
    it. The audio is downloaded for this, so its path is returned for the
    transcription to reuse instead of downloading it a second time.
    """
    audio_path = fetch.download_audio(video.url, cfg.paths.audio_dir, cfg)
    try:
        # Always sampled with the Bangla model: what it writes is the signal.
        text = pool.for_language(langs.BANGLA).sample_text(audio_path, langs.BANGLA)
        detection = langdetect.classify(text, cfg.asr.auto_english_below, cfg.asr.default_language)
        if not detection.confident:
            logger.warning(
                "%s: too little speech to detect the language (%d words); using %r. "
                "Pass --lang to choose.", video.id, detection.words, detection.language,
            )
        else:
            logger.info(
                "%s: %.1f%% Bangla function words in %d sampled words -> %s",
                video.id, 100 * detection.rate, detection.words, detection.language,
            )
        store.set_video_language(conn, video.id, detection.language)
    except BaseException:
        audio_path.unlink(missing_ok=True)
        raise
    return audio_path


def transcribe_one(
    conn, cfg: Config, pool: TranscriberPool, video: store.Video, audio_path: Path | None = None
) -> None:
    """Runs stage 1 for a single video: fetch metadata, prefer human subtitles
    in the video's language over ASR, store segments, write exports, delete audio.
    `audio_path` is audio that is already downloaded (see `detect_one`).
    """
    info = fetch.get_video_info(video.url, cfg)
    store.upsert_video(
        conn,
        store.Video(
            id=video.id,
            platform=info.platform,
            url=info.url,
            title=info.title,
            channel=info.channel,
            channel_id=info.channel_id,
            upload_date=info.upload_date,
            duration=info.duration,
            description=info.description,
        ),
    )

    language = video.language or cfg.asr.default_language
    store.set_video_language(conn, video.id, language)

    try:
        sub_url = None if cfg.fetch.force_asr else info.human_subtitles.get(language)
        if sub_url:
            segments = fetch.fetch_human_subtitle_segments(sub_url)
            source, asr_model = "human_subs", None
        else:
            transcriber = pool.for_language(language)  # fails first if no model handles the language
            if audio_path is None:
                audio_path = fetch.download_audio(info.url, cfg.paths.audio_dir, cfg)
            result = transcriber.transcribe(audio_path, language)
            segments = result.segments
            _warn_on_suspect_asr(info, result.stats)
            source, asr_model = "asr", transcriber.model_cfg.key
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)

    store.replace_segments(conn, video.id, segments)
    store.set_video_source(conn, video.id, source, asr_model)
    updated = store.get_video(conn, video.id)
    assert updated is not None
    assessment = write_transcript_files(cfg, updated, segments)
    for flag in assessment.flags:
        logger.warning("%s: %s", video.id, flag)
    store.set_video_status(conn, video.id, "transcribed")

    # Best-effort extras: the transcript is already safe in the DB, so a
    # locked file or over-long path here must not mark the video failed.
    try:
        write_shared_copy(cfg, updated, segments, assessment)
    except Exception:  # noqa: BLE001
        logger.warning("Could not write transcript copy for %s", video.id, exc_info=True)
    try:
        ledger.append_entry(ledger.ledger_path(cfg), updated)
    except Exception:  # noqa: BLE001
        logger.warning("Could not update transcription ledger for %s", video.id, exc_info=True)


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
        pool = TranscriberPool(cfg)
        # Audio downloaded while detecting a language, kept for the transcription.
        detected_audio: dict[str, Path] = {}

        language_undecided = [v for v in queued if v.language is None]
        if language_undecided:
            _run_stage(
                conn, language_undecided, "detect language",
                lambda v: detected_audio.__setitem__(v.id, detect_one(conn, cfg, pool, v)),
                result, "detected",
            )
            queued = [v for v in (store.get_video(conn, v.id) for v in queued) if v and v.status == "queued"]

        # Group by language (stable, so queue order holds within a language) so
        # each ASR model is loaded once instead of once per switch.
        queued.sort(key=lambda v: v.language or cfg.asr.default_language)
        try:
            _run_stage(
                conn, queued, "transcribe",
                lambda v: transcribe_one(conn, cfg, pool, v, audio_path=detected_audio.get(v.id)),
                result, "transcribed",
            )
        finally:
            # transcribe_one deletes its own audio; this catches early failures
            for path in detected_audio.values():
                path.unlink(missing_ok=True)
            pool.unload()

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


def reexport(conn, cfg: Config, video_ids: list[str] | None = None) -> list[str]:
    """Rewrites the transcript files (data/videos/<id>/ and the shared copy,
    plus INDEX.md) of already transcribed videos from the database, without
    running ASR again. Picks up the current layout, glossary and quality
    thresholds. `video_ids` limits it to those videos (unknown or untranscribed
    ids are skipped). Returns the ids that were rewritten."""
    if video_ids is None:
        videos = [v for v in store.list_videos(conn) if v.status != "queued"]
    else:
        videos = [v for v in (store.get_video(conn, i) for i in video_ids) if v is not None]
    done: list[str] = []
    for video in videos:
        segments = store.get_segments(conn, video.id)
        if not segments:
            continue
        assessment = write_transcript_files(cfg, video, segments, summary=store.get_summary(conn, video.id))
        write_shared_copy(cfg, video, segments, assessment)
        done.append(video.id)
    return done


def retry_failed(conn) -> int:
    """Requeues failed videos at the earliest stage that still has work to
    do, so finished stages (e.g. expensive ASR) are not repeated."""
    failed = store.list_videos(conn, status="failed")
    for v in failed:
        if store.get_chunks(conn, v.id):
            status = "indexed"
        elif store.get_segments(conn, v.id):
            status = "transcribed"
        else:
            status = "queued"
        store.set_video_status(conn, v.id, status, error=None)
    return len(failed)
