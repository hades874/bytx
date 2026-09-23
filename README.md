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
   `summary.md`.
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
  project was built and tested on a GTX 1650 (4GB VRAM).
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

### A note on speed (GTX 1650 / similar low-end GPUs)

GTX 16xx-series cards do **not** have Tensor Cores (unlike RTX cards), so
fp16 doesn't get the usual speedup on them, and `beam_size` directly
multiplies decoding cost — this is not the "4x faster than real-time"
faster-whisper is often known for on cards with Tensor Cores. Measured on a
GTX 1650 with whisper-medium (tugstugi-medium-bn, fp16):

| Setting | RTF (time / audio duration) |
|---|---|
| `beam_size=1`, default `repetition_penalty`/`no_repeat_ngram_size` off | **~0.9** (roughly real-time) |
| `beam_size=5` | 5-10x+ slower |
| `beam_size=1` **with** `repetition_penalty≠1.0` or `no_repeat_ngram_size≠0` | 5-10x+ slower |

The last row is a real faster-whisper/CTranslate2 quirk: those two options
force a much slower per-token CPU↔GPU score-adjustment path. `config.toml`
ships with them off by default and relies on
`transcribe.collapse_repeated_ngrams()` (a cheap post-processing pass) to
clean up hallucinated repeats instead. **Don't re-enable them without
re-checking speed.**

**Open caveat, tested and not yet fully explained:** the ~0.9 RTF above was
measured on a short (15s), single-VAD-segment clip. On a full ~5-minute
real video with the shipped defaults, a run took well over 30 minutes
(RTF > 9), even though nothing else changed. The GPU was confirmed
genuinely busy throughout (97-99% via `nvidia-smi`), not hung. The most
likely explanation is that Whisper internally processes audio in fixed
30-second windows regardless of VAD segment length, and cost may not scale
as linearly with segment count/length on this GPU as the short-clip number
suggests -- but this wasn't conclusively isolated. **Practical takeaway:**
on a GTX 1650 (or similar Tensor-Core-less card), budget for a full video
to take substantially longer than its own duration -- possibly 5-10x+, not
the ~1x the short-clip test would suggest. Treat `bytx run` as a
batch/overnight job. If this matters to you, worth trying before assuming
it's fixed:
- A smaller model (e.g. convert and `bench` against Whisper-small) --
  untested here, but the encoder pass is the likely fixed cost, and
  whisper-medium's encoder is meaningfully heavier than small's.
- `device = "cpu"` in `config.toml`, in case CPU multi-threading actually
  beats this particular GPU for this model -- untested here.
- Filing/searching a faster-whisper issue with a real long recording, since
  this may be a known scaling characteristic rather than something
  specific to this setup.

Practical guidance:
- The shipped defaults (`beam_size=1`, penalties off) give roughly
  real-time transcription on a GTX 1650. Raise `beam_size` only if you
  have a Tensor-Core GPU (RTX-series) or don't mind a much longer batch
  run for the accuracy gain.
- `bytx doctor` reports the actual device/compute_type in use, and
  `Transcriber` automatically falls back (fp16 → int8_float16 → int8 →
  CPU) if a compute_type doesn't work at all on your card.

## Project layout

```
src/bytx/
  cli.py          command-line interface
  config.py       config.toml loader
  fetch.py        yt-dlp: URL expansion, metadata, audio download, subtitles
  transcribe.py   faster-whisper wrapper (VAD, fallback chain, anti-hallucination)
  store.py        SQLite schema + FTS5 (trigram) search
  export.py       txt/srt/vtt/json writers
  pipeline.py     staged, resumable runner (transcribe -> index -> summarize)
  llm.py          minimal Ollama HTTP client
  rag.py          chunking, embeddings, semantic search, ask
  summarize.py    map-reduce summarization
  bench.py        model comparison (speed + WER/CER)
  cuda_dll.py     Windows CUDA/cuDNN DLL path fix (see below)
scripts/
  convert_model.py   HF Whisper -> CTranslate2 conversion
tests/               offline pytest suite (no GPU/network required)
```

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
