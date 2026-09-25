# bytx

A local, free/open-source tool to transcribe Bangla-language YouTube videos and
build a searchable, askable library out of them. No paid APIs: speech
recognition runs on your own GPU/CPU via [faster-whisper][fw], and
summarization/Q&A run on your own [Ollama][ollama] install.

[fw]: https://github.com/SYSTRAN/faster-whisper
[ollama]: https://ollama.com

## What it does

1. **Fetch** — takes a single video URL, a playlist, a whole channel, or a
   text file of URLs (`bytx add`). Uses uploader-made (human) Bangla
   subtitles when a video has them; otherwise transcribes the audio with a
   Bangla-tuned Whisper model.
2. **Store** — everything lands in one SQLite database (`data/library.db`)
   plus a readable folder per video (`data/videos/<id>/`) with
   `transcript.txt` / `.srt` / `.vtt` / `.json`, `info.json`, and
   `summary.md`. A second, human-browsable copy of each plain-text
   transcript also lands in the git-tracked `yt-transcribed-files/` folder
   (one subfolder per video, named `<channel> - <title>`), and every
   completed video is recorded in the git-tracked `TRANSCRIBED_LEDGER.md`
   so a video is never silently re-transcribed, even if the gitignored
   `data/` folder is deleted or the repo is cloned fresh.
3. **Search** — keyword search (`bytx search`) matches word *parts*, so a
   search for a root word also finds it with Bangla suffixes attached (e.g.
   বাংলাদেশ finds বাংলাদেশের). `--semantic` searches by meaning instead,
   using local embeddings.
4. **Ask** — `bytx ask "<question>"` answers from your transcript library
   using your local Ollama chat model, with timestamped citations back to
   the source video.
5. **Summarize** — `bytx summarize` (or automatically during `bytx run`)
   produces a Bangla summary per video via map-reduce over the transcript.

## Requirements

- Windows with an NVIDIA GPU (CPU-only also works, just slower) — this
  project runs on an RTX 4070 (12GB VRAM).
- [Ollama](https://ollama.com) installed and running.
- [uv](https://docs.astral.sh/uv/) for Python package management.
- ffmpeg and [Deno](https://deno.com) (yt-dlp needs a JS runtime for
  YouTube). Install with:
  ```
  winget install Gyan.FFmpeg DenoLand.Deno
  ```

## Setup

```
uv sync
uv run bytx setup      # converts the ASR model, pulls the Ollama models
uv run bytx doctor      # checks everything is wired up correctly
```

`bytx setup` needs the `convert` extra (transformers + torch, CPU-only, used
only for the one-time model conversion) the first time it runs a new model:

```
uv sync --extra convert
```

## Usage

```
# Queue and process a single video
uv run bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --run

# Queue a whole channel or playlist without processing yet
uv run bytx add "https://www.youtube.com/@SomeChannel"
uv run bytx run                       # process everything queued

# Queue from a text file (one URL per line, '#' comments allowed)
uv run bytx add urls.txt --run

# Check on things
uv run bytx status
uv run bytx list --status transcribed

# Read / export a transcript
uv run bytx show <video_id>
uv run bytx export <video_id> --format srt --output out.srt

# Search
uv run bytx search "বাংলাদেশ"
uv run bytx search "বাংলাদেশের ইতিহাস" --semantic

# Ask questions across your library (or one video with --video)
uv run bytx ask "এই ভিডিওতে কী বলা হয়েছে?"

# Retry anything that failed
uv run bytx retry && uv run bytx run
```

The pipeline is resumable: if it's interrupted (Ctrl+C, crash, out of
disk), just run `bytx run` again — already-finished videos are skipped and
nothing is redone.

## Configuration

All settings live in `config.toml` at the project root: which ASR model to
use, VAD/decoding parameters, which Ollama models to use for embeddings and
chat, chunk sizes for retrieval, and yt-dlp options (cookies, sleep
intervals). See the comments in that file.

### How audio is cut into segments

Whisper stops generating after **224 tokens per 30 s window**, and Bengali
needs about 0.7 tokens per character. A fast or dense 30 s window therefore
overflows: the text is cut off mid-sentence (sometimes mid-word) and the rest
of that window is silently lost. Early transcripts were affected (94% of
segments sat exactly at the cap; on one fast speaker ~40% of the words were
missing).

To avoid this, `Transcriber.transcribe()` does its own segmentation instead of
handing the whole file to faster-whisper:

1. Silero VAD finds speech, and adjacent speech is merged into clips of at most
   `asr.max_clip_seconds` (12 s) at natural pauses.
2. Each clip is decoded separately, so every segment is one phrase-sized clip
   with real start/end times.
3. Any clip whose text comes within 8 tokens of the limit is split at its
   quietest point and re-decoded. Clips that still hit the limit at minimum
   size are counted in `TranscribeResult.stats.truncated_unresolved`.

`bytx run` logs a warning when clips are still truncated, or when less than
75% of a video was detected as speech (a lot of music/silence, or a too-high
`asr.vad_threshold`). `bytx bench` prints the clip/re-split counts too.
Transcripts made before this change are not regenerated.

### Known limitations of the transcripts

These come from the ASR model, not the segmentation:

- Numbers are written as spoken words ("দুই হাজার সাতাইশ"), not digits.
- English words are transliterated into Bengali script and are often
  misheard ("এস অপি" for SOP, "স্টোর" for story).
- No punctuation; segment breaks are speech pauses, not sentences.
- Music/outro sections that slip past VAD can produce nonsense text.

### Choosing an ASR model

Two models are pre-configured:

- `tugstugi-medium-bn` (default) — a Bangla fine-tune of Whisper-medium.
  Best accuracy on Bangla speech.
- `large-v3-turbo` — multilingual; may handle English words mixed into
  Bangla speech ("Banglish") better.

Compare them on your own content before committing to one:

```
uv run --extra convert python scripts/convert_model.py large-v3-turbo
uv run bytx bench "<url>" --clip 01:00-03:00 --models tugstugi-medium-bn,large-v3-turbo
```

With a reference transcript (`--ref reference.txt`), `bench` also reports
WER/CER.

### A note on speed (RTX 4070)

This project runs on an **RTX 4070** (Tensor Cores, 12GB VRAM). `bytx doctor`
reports `device=cuda compute_type=float16`. During a batch run the GPU sits
at roughly 40% utilization with ~3GB VRAM used by the medium model
(tugstugi-medium-bn, fp16), since download, VAD and CPU-side decoding leave
the GPU idle part of the time. No real-time factor (RTF, time / audio
duration) has been measured on this card yet; use `bytx bench` to measure it
rather than assuming a number.

Two settings affect decoding speed:

- `beam_size` directly multiplies decoding cost. `config.toml` ships with
  `beam_size=1`; raise it only if you want the accuracy gain and don't mind a
  slower batch run.
- `repetition_penalty≠1.0` or `no_repeat_ngram_size≠0` is a real
  faster-whisper/CTranslate2 quirk: those two options force a much slower
  per-token CPU↔GPU score-adjustment path. `config.toml` ships with them off
  and relies on `transcribe.collapse_repeated_ngrams()` (a cheap
  post-processing pass) to clean up hallucinated repeats instead. **Don't
  re-enable them without re-checking speed.**

Treat `bytx run` as a batch job. If speed matters, options to try:
- A smaller model (e.g. convert and `bench` against Whisper-small).
- `device = "cpu"` in `config.toml` (untested here, unlikely to beat the GPU).

`bytx doctor` reports the actual device/compute_type in use, and
`Transcriber` automatically falls back (fp16 → int8_float16 → int8 → CPU) if
a compute_type doesn't work at all on your card.

## Project layout

```
src/bytx/
  cli.py          command-line interface
  config.py       config.toml loader
  fetch.py        yt-dlp: URL expansion, metadata, audio download, subtitles
  transcribe.py   faster-whisper wrapper (VAD, fallback chain, anti-hallucination)
  store.py        SQLite schema + FTS5 (trigram) search
  export.py       txt/srt/vtt/json writers
  ledger.py       durable, git-tracked markdown record of transcribed video ids
  pipeline.py     staged, resumable runner (transcribe -> index -> summarize)
  llm.py          minimal Ollama HTTP client
  rag.py          chunking, embeddings, semantic search, ask
  summarize.py    map-reduce summarization
  bench.py        model comparison (speed + WER/CER)
  cuda_dll.py     Windows CUDA/cuDNN DLL path fix (see below)
scripts/
  convert_model.py   HF Whisper -> CTranslate2 conversion
tests/               offline pytest suite (no GPU/network required)
TRANSCRIBED_LEDGER.md   git-tracked list of transcribed video ids (see below)
yt-transcribed-files/   git-tracked "<channel> - <title>" transcript folders
```

### Transcribed-video ledger and human-facing output

`data/` (SQLite DB + per-id export folders) is gitignored, so on its own it
can't tell a fresh clone, or a wiped `data/`, that a video has already been
transcribed. Two things fix that:

- **`TRANSCRIBED_LEDGER.md`** — a plain, append-only markdown record (video
  id, channel, title, url, date) written after every successful
  transcription. `bytx add` checks it alongside the DB, so a known video id
  is never silently re-queued.
- **`yt-transcribed-files/<channel> - <title>/transcript.txt`** — a
  human-browsable, git-tracked copy of the plain-text transcript, separate
  from the DB-linked `data/videos/<id>/` export.

To force a video to be re-transcribed, remove its entry from
`TRANSCRIBED_LEDGER.md` (and its row from `data/library.db`, if present).

## Known Windows quirk (already handled)

Pip-installed `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels put their DLLs
under `site-packages/nvidia/*/bin`, which Windows never searches by default.
`os.add_dll_directory()` alone isn't enough — CTranslate2's own DLL loader
only respects the `PATH` environment variable. `cuda_dll.py` handles both
mechanisms; you shouldn't need to think about this, but if you ever see
`cublas64_12.dll is not found or cannot be loaded`, that's the symptom.

## Development

```
uv run pytest              # full test suite (offline, no GPU/network needed)
uv run ruff check .        # lint
```
