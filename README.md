# bytx

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#requirements)

A local, free/open-source tool to transcribe Bangla- and English-language
YouTube and Facebook videos into a searchable, askable library. No paid
APIs: speech recognition runs on your own GPU/CPU via [faster-whisper][fw],
and summarization/Q&A run on your own [Ollama][ollama] install.

[fw]: https://github.com/SYSTRAN/faster-whisper
[ollama]: https://ollama.com

## Features

- **Fetch** a single video, a playlist, a whole channel, or a text file of
  URLs (YouTube or Facebook). Uses the uploader's own subtitles when
  available (YouTube only), otherwise transcribes the audio.
- **Bilingual, on purpose** — every video is tagged `bn` or `en` by you
  (`--lang`), which picks the matching Whisper model. Nothing is
  auto-detected from Whisper's language ID or YouTube's metadata, since both
  mislabel Bangla speech that mixes in English words. See [Languages](#languages).
- **Store** everything in a local SQLite database plus readable
  `.txt`/`.srt`/`.vtt`/`.json` exports per video.
- **Search** by keyword (FTS5, Bangla-suffix aware) or by meaning
  (`--semantic`, local multilingual embeddings).
- **Ask** natural-language questions across your library with a local
  Ollama chat model, answered with timestamped citations.
- **Summarize** every video via map-reduce, in its own language.
- **Quality flags and cross-checks** — words/min, gaps, and low-confidence
  clips are flagged automatically; `bytx compare` cross-checks a transcript
  against YouTube's auto-captions. See [Trusting a transcript](#trusting-a-transcript).

## Requirements

- Windows with an NVIDIA GPU (CPU-only also works, just slower).
- [Ollama](https://ollama.com) installed and running.
- [uv](https://docs.astral.sh/uv/) for Python package management.
- ffmpeg and [Deno](https://deno.com) (yt-dlp needs a JS runtime for YouTube):
  ```
  winget install Gyan.FFmpeg DenoLand.Deno
  ```

## Setup

```
uv sync
uv run bytx setup      # converts/downloads every ASR model, pulls the Ollama models
uv run bytx doctor      # checks everything is wired up correctly
```

`bytx setup` needs the `convert` extra (transformers + torch, CPU-only, used
only for the one-time model conversion) the first time it runs a new model:

```
uv sync --extra convert
```

## Usage

```
# Queue and process a single video (Bangla by default)
uv run bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --run

# English video, playlist, channel or list of URLs: say so with --lang
uv run bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --lang en --run
uv run bytx add "https://www.youtube.com/@SomeEnglishChannel" --lang en
uv run bytx add english-urls.txt --lang en

# A mixed list, and you don't know which is which: decide per video
uv run bytx add mixed-urls.txt --lang auto

# Queue a whole channel or playlist without processing yet
uv run bytx add "https://www.youtube.com/@SomeChannel"
uv run bytx run                       # process everything queued

# Facebook videos work the same way, and can share a file with YouTube ones
uv run bytx add "https://www.facebook.com/watch/?v=123456789012345" --run

# Check on things
uv run bytx status
uv run bytx list --status transcribed --lang en --platform facebook

# Read / export a transcript
uv run bytx show <video_id>
uv run bytx export <video_id> --format srt --output out.srt

# Search
uv run bytx search "বাংলাদেশ"
uv run bytx search "student visa" --semantic

# Ask questions across your library (or one video with --video)
uv run bytx ask "এই ভিডিওতে কী বলা হয়েছে?"
uv run bytx ask "What documents do I need for the visa?"

# How far can a transcript be trusted?
uv run bytx quality                    # every transcript, flagged ones first
uv run bytx compare <video_id>         # cross-check against YouTube's auto-captions

# Retry anything that failed
uv run bytx retry && uv run bytx run
```

The pipeline is resumable: if it's interrupted (Ctrl+C, crash, out of
disk), just run `bytx run` again — already-finished videos are skipped.

## Configuration

All settings live in `config.toml` at the project root: which ASR model
serves each language, VAD/decoding parameters, Ollama models for embeddings
and chat, chunk sizes for retrieval, and yt-dlp options (cookies, sleep
intervals). See the comments in that file for details.

### Languages

Every video is tagged `bn` or `en`, which picks the ASR model
(`[asr.language_models]` in `config.toml`) and which uploader subtitles are
accepted. You choose it with `bytx add --lang`; without it, the default is
`asr.default_language` (`bn`).

Nothing guesses a video's language unless you ask for it (`--lang auto`).
Whisper's own language ID and YouTube's language metadata both frequently
mislabel Bangla videos that mix in English words as English or Hindi, so
neither is used. `--lang auto` instead transcribes a short audio sample with
the Bangla model and counts Bangla function words to decide — it only
distinguishes Bangla from English (a Hindi video would be called English),
so use an explicit `--lang` when you know it.

### Facebook

Facebook support goes through yt-dlp, no extra install needed.

- Single videos and reels work directly; share links (`fb.watch/…`) only
  work if yt-dlp can follow the redirect without hitting a login wall.
- No Facebook page/playlist extractor in yt-dlp yet — add individual video
  URLs instead.
- Private/friends-only/group videos need `cookies_from_browser` set under
  `[fetch]` in `config.toml`.
- No uploader subtitles, so these always go through ASR.
- Ids are stored as `fb_<number>` (vs. YouTube's bare 11-character id).

### Trusting a transcript

A Whisper transcript is good enough to find leads and not good enough to
treat as a source: numbers usually survive, but names and acronyms get
garbled, and stretches of speech can silently go missing.

**Quality flags** (`bytx quality`), computed from the stored segments:

| Signal | Flag when | Usually means |
|---|---|---|
| words per minute | under 70 or over 220 | speech was skipped, or text was repeated/invented |
| gaps | a stretch over 10 s with no text | dropped speech, or music |
| low-confidence clips | low `avg_logprob` or high `no_speech_prob` | a garbled or hallucinated clip |

These are smoke alarms, not proof — a flagged transcript may be fine, and an
unflagged one can still contain a wrong name.

**The glossary** (`glossary.txt`) lists names/acronyms/jargon your videos
use, in `canonical <= known mis-hearing` format. It primes the decoder (for
languages in `asr.glossary_prompt_languages`) and produces a corrected
`transcript.clean.txt` alongside the untouched raw transcript. Run `bytx
reexport --all` after editing it to refresh existing transcripts.

**`bytx compare <video_id>`** cross-checks a transcript against YouTube's
own auto-captions; segments where they disagree are worth a listen.

Known limitations: Bangla numbers are written as spoken words, not digits;
English words inside Bangla speech are often misheard; Bangla has no
punctuation; a video is decoded entirely in the one language it was
assigned, so a video that switches languages mid-way comes out partly
garbled; Hindi is not supported.

## Project layout

```
src/bytx/
  cli.py          command-line interface
  config.py       config.toml loader
  fetch.py        yt-dlp: URL expansion, metadata, audio download, subtitles
  transcribe.py   faster-whisper wrapper (VAD clipping, per-clip confidence, glossary priming)
  store.py        SQLite schema + FTS5 search
  export.py       txt/srt/vtt/json writers, metadata.json, INDEX.md
  quality.py      words/min, coverage, gap and low-confidence flags
  glossary.py     decoder priming terms and cleaned-copy corrections
  compare.py      cross-check against YouTube auto-captions
  ledger.py       append-only record of transcribed video ids
  pipeline.py     staged, resumable runner (transcribe -> index -> summarize)
  rag.py          chunking, embeddings, semantic search, ask
  summarize.py    map-reduce summarization
  bench.py        model comparison (speed + WER/CER)
  cuda_dll.py     Windows CUDA/cuDNN DLL path fix
scripts/
  convert_model.py   HF Whisper -> CTranslate2 conversion
tests/               offline pytest suite (no GPU/network required)
```

`TRANSCRIBED_LEDGER.md` and `yt-transcribed-files/` are your personal
transcript library (append-only ledger + human-browsable exports) — both
gitignored by default, trackable if you want a git-backed archive of your
own.

## Known Windows quirk (already handled)

Pip-installed `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` wheels put their DLLs
where Windows doesn't search by default. `cuda_dll.py` fixes this; if you
ever see `cublas64_12.dll is not found or cannot be loaded`, that's the
symptom it handles.

## Development

```
uv run pytest              # full test suite (offline, no GPU/network needed)
uv run ruff check .        # lint
```

Contributions and issues are welcome — open a PR or an issue.

## License

[MIT](LICENSE)
