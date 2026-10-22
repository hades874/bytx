# bytx

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](#install)

A local, free/open-source tool to transcribe Bangla- and English-language
YouTube and Facebook videos into a searchable, askable library. No paid
APIs and nothing leaves your machine: speech recognition runs on your own
GPU or CPU via [faster-whisper][fw], and the optional summaries and Q&A run
on your own [Ollama][ollama] install.

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
- **Search** by keyword (FTS5, Bangla-suffix aware) or, with Ollama, by
  meaning (`--semantic`, local multilingual embeddings).
- **Ask** natural-language questions across your library with a local
  Ollama chat model, answered with timestamped citations (needs Ollama).
- **Summarize** every video via map-reduce, in its own language (needs Ollama).
- **Quality flags and cross-checks** — words/min, gaps, and low-confidence
  clips are flagged automatically; `bytx compare` cross-checks a transcript
  against YouTube's auto-captions. See [Trusting a transcript](#trusting-a-transcript).

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It
fetches Python 3.12 for you if you don't have it. Nothing else: yt-dlp brings
the JavaScript runtime YouTube needs (Deno), and audio is decoded without
ffmpeg.

```
uv tool install bytx              # any OS; transcribes on the CPU
uv tool install "bytx[gpu]"       # Windows or Linux with an NVIDIA GPU: adds the CUDA libraries (about 1.3 GB)

bytx setup                        # downloads the speech models once
bytx doctor                       # checks everything is ready
bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --lang bn --run
```

`bytx setup` downloads the Bangla model (about 1.5 GB) and the English model
(about 1.6 GB) from Hugging Face. If [Ollama][ollama] is installed and running
it also pulls its two models (about 5 GB); otherwise it skips them.

Upgrade with `uv tool upgrade bytx`. Your library and settings are kept.

### Ollama is optional

Transcription, keyword search, export and the quality checks work without
Ollama. Search by meaning, Q&A and summaries need it: without it, `bytx run`
stops after transcribing (the videos stay `transcribed`, not `failed`) and
says so. Install Ollama later, run `bytx setup` to pull its models, then
`bytx run` to index and summarize what you already have.

### Speed

On an NVIDIA GPU a video transcribes in a fraction of its length. On a CPU
expect about **2.25× the video's length** on a 6-core desktop CPU (Ryzen 5
5600), and slower on laptops: a 30-minute video can take well over an hour.
`bytx doctor` tells you if you have an NVIDIA GPU that the CPU-only install
isn't using.

- **macOS:** CPU only (there is no CUDA on macOS).
- **Linux with an NVIDIA GPU** (community-tested): if transcription falls
  back to the CPU with a `libcublas`/`libcudnn` error, point the loader at the
  CUDA libraries the `gpu` extra installed:
  ```
  BYTX_PY="$(uv tool dir)/bytx/bin/python"
  export LD_LIBRARY_PATH="$("$BYTX_PY" -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')"
  ```

## Usage

```
# Queue and process a single video
bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --lang bn --run

# English video, playlist, channel or list of URLs: say so with --lang
bytx add "https://www.youtube.com/watch?v=XXXXXXXXXXX" --lang en --run
bytx add "https://www.youtube.com/@SomeEnglishChannel" --lang en
bytx add english-urls.txt --lang en

# A mixed list, and you don't know which is which: decide per video
bytx add mixed-urls.txt --lang auto

# Queue a whole channel or playlist without processing yet
bytx add "https://www.youtube.com/@SomeChannel" --lang bn
bytx run                              # process everything queued

# Facebook videos work the same way, and can share a file with YouTube ones
bytx add "https://www.facebook.com/watch/?v=123456789012345" --lang bn --run

# Check on things
bytx status
bytx list --status transcribed --lang en --platform facebook

# Read / export a transcript
bytx show <video_id>
bytx export <video_id> --format srt --output out.srt

# Search
bytx search "বাংলাদেশ"
bytx search "student visa" --semantic         # needs Ollama

# Ask questions across your library (or one video with --video); needs Ollama
bytx ask "এই ভিডিওতে কী বলা হয়েছে?"
bytx ask "What documents do I need for the visa?"

# How far can a transcript be trusted?
bytx quality                          # every transcript, flagged ones first
bytx compare <video_id>               # cross-check against YouTube's auto-captions

# Retry anything that failed
bytx retry && bytx run

# Queued it with the wrong language? Remove it completely, then add it again
bytx remove <video_id>                # dry run: lists what would be deleted
bytx remove <video_id> --yes
```

The pipeline is resumable: if it's interrupted (Ctrl+C, crash, out of
disk), just run `bytx run` again — already-finished videos are skipped.

## Your data folder

Everything bytx keeps about your library lives in one folder, `~/bytx`
(on Windows `C:\Users\<you>\bytx`), created the first time you run a
command. Set the `BYTX_HOME` environment variable to keep it somewhere else.

```
~/bytx/
  config.toml              your settings (optional; only what you change)
  data/library.db          the database: transcripts, search index, summaries
  data/videos/<id>/        per-video exports (txt, srt, vtt, json, summary)
  yt-transcribed-files/    readable copies, "<date>/<channel> - <title>/", plus INDEX.md
  TRANSCRIBED_LEDGER.md    record of transcribed videos; `bytx add` skips them
  glossary.txt             your vocabulary (optional, see below)
```

The speech models are not stored there: they live in the Hugging Face cache
(`~/.cache/huggingface`; move it with `HF_HOME`). Ollama keeps its own models
(move them with `OLLAMA_MODELS`).

## Configuration

bytx ships with sensible defaults
([`src/bytx/defaults.toml`](src/bytx/defaults.toml), with comments on every
setting): which ASR model serves each language, VAD/decoding parameters,
quality thresholds, Ollama models, retrieval chunk sizes and yt-dlp options.
To change one, put just that setting in `config.toml` in your data folder.
Tables merge with the defaults and anything else replaces them, so new
defaults in a later release still reach you. For example:

```toml
[asr]
device = "cpu"                 # never try the GPU

[fetch]
cookies_from_browser = "firefox"   # for age-gated or private videos you can see when logged in
```

`--config <path>` uses another settings file instead; its folder then
becomes the data folder.

### Languages

Every video is tagged `bn` or `en`, which picks the ASR model
(`[asr.language_models]`) and which uploader subtitles are accepted. You
choose it with `bytx add --lang`; without it, the default is
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
  `[fetch]` in your `config.toml`.
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

**`bytx compare <video_id>`** cross-checks a transcript against YouTube's
own auto-captions; segments where they disagree are worth a listen.

Known limitations: Bangla numbers are written as spoken words, not digits;
English words inside Bangla speech are often misheard; Bangla has no
punctuation; a video is decoded entirely in the one language it was
assigned, so a video that switches languages mid-way comes out partly
garbled; Hindi is not supported.

### The glossary

`glossary.txt` in your data folder lists names, acronyms and jargon your
videos use, one entry per line (`#` starts a comment):

```
IRCC                                   a term to prime the decoder with
TrustEd Ireland <= Trusted Ireland     canonical <= known mis-hearing | another mis-hearing
TCD <= ECD   # only if you never cover ECD (early childhood development)
```

Every term is fed to Whisper as a hint for the languages in
`asr.glossary_prompt_languages` (English only by default; priming the Bangla
model can make it echo the prompt), so keep the list to words that matter.
The `<=` variants are corrected only in a cleaned copy of each transcript
(`transcript.clean.txt`); the raw transcript is never changed. Matching is
case-insensitive, and whole-word for Latin text. After editing it, run
`bytx reexport --all` to refresh existing transcripts.

## Responsible use

bytx downloads a video's audio to transcribe it on your own computer. Only
transcribe videos you have the right to use, respect copyright and the
terms of service of YouTube and Facebook, and don't republish other
people's words without permission. If you set `cookies_from_browser`, bytx
acts with your own logged-in account, so the same rules apply to you.

## Project layout

```
src/bytx/
  cli.py          command-line interface
  config.py       packaged defaults + your config.toml
  defaults.toml   the default settings
  models.py       where the ASR models are (Hugging Face cache), downloading them
  fetch.py        yt-dlp: URL expansion, metadata, audio download, subtitles
  transcribe.py   faster-whisper wrapper (VAD clipping, per-clip confidence, glossary priming)
  store.py        SQLite schema + FTS5 search
  export.py       txt/srt/vtt/json writers, metadata.json, INDEX.md
  quality.py      words/min, coverage, gap and low-confidence flags
  glossary.py     decoder priming terms and cleaned-copy corrections
  compare.py      cross-check against YouTube auto-captions
  ledger.py       append-only record of transcribed video ids
  pipeline.py     staged, resumable runner (transcribe -> index -> summarize), remove
  rag.py          chunking, embeddings, semantic search, ask
  summarize.py    map-reduce summarization
  bench.py        model comparison (speed + WER/CER)
  cuda_dll.py     Windows CUDA/cuDNN DLL path fix
scripts/
  convert_model.py   maintainer tool: Hugging Face Whisper -> CTranslate2
tests/               offline pytest suite (no GPU/network required)
```

## Known Windows quirk (already handled)

The CUDA libraries from the `gpu` extra (`nvidia-cublas-cu12`,
`nvidia-cudnn-cu12`) put their DLLs where Windows doesn't search by default.
`cuda_dll.py` fixes this; if you ever see `cublas64_12.dll is not found or
cannot be loaded`, that's the symptom it handles.

## Development

```
git clone https://github.com/hades874/bytx && cd bytx
uv sync --extra gpu        # or plain `uv sync` without an NVIDIA GPU
uv run pytest              # full test suite (offline, no GPU/network needed)
uv run ruff check .        # lint

uv tool install --editable ".[gpu]"   # a `bytx` on PATH that runs your working copy
```

`bytx bench --clip` compares models on part of a video and needs
[ffmpeg](https://ffmpeg.org/download.html) on PATH (nothing else in bytx
does). `scripts/convert_model.py` converts a Hugging Face Whisper checkpoint
to CTranslate2; it needs `uv sync --extra convert` (transformers + torch).

Contributions and issues are welcome — open a PR or an issue.

## License

[MIT](LICENSE)
