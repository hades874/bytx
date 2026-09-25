# bytx

A local, free/open-source tool to transcribe Bangla- and English-language
YouTube and Facebook videos and build a searchable, askable library out of them. Each
video has a language (Bangla unless you say `--lang en` or `--lang auto`)
that picks the right speech model. No paid APIs: speech
recognition runs on your own GPU/CPU via [faster-whisper][fw], and
summarization/Q&A run on your own [Ollama][ollama] install.

[fw]: https://github.com/SYSTRAN/faster-whisper
[ollama]: https://ollama.com

## What it does

1. **Fetch** — takes a single video URL (YouTube or Facebook), a playlist, a
   whole channel, or a text file of URLs (`bytx add`). Uses uploader-made
   (human) subtitles in the video's language when it has them (YouTube only;
   Facebook videos always go through ASR); otherwise transcribes the audio
   with a Bangla-tuned Whisper model (Bangla) or Whisper large-v3-turbo
   (English). See [Languages](#languages).
2. **Store** — everything lands in one SQLite database (`data/library.db`)
   plus a readable folder per video (`data/videos/<id>/`) with
   `transcript.txt` / `.srt` / `.vtt` / `.json`, `info.json`, and
   `summary.md`. A second, human-browsable copy of each plain-text
   transcript also lands in the git-tracked `yt-transcribed-files/` folder
   (grouped in a `YYYY-MM-DD` folder for the day it was transcribed, then
   one subfolder per video, named `<channel> - <title>`), and every
   completed video is recorded in the git-tracked `TRANSCRIBED_LEDGER.md`
   so a video is never silently re-transcribed, even if the gitignored
   `data/` folder is deleted or the repo is cloned fresh.
3. **Search** — keyword search (`bytx search`) matches word *parts*, so a
   search for a root word also finds it with Bangla suffixes attached (e.g.
   বাংলাদেশ finds বাংলাদেশের), and is case-insensitive for English.
   `--semantic` searches by meaning instead, using local embeddings (the
   `bge-m3` model is multilingual, so one question can find both languages).
4. **Ask** — `bytx ask "<question>"` answers from your transcript library
   using your local Ollama chat model, with timestamped citations back to
   the source video. It answers in the language you ask in.
5. **Summarize** — `bytx summarize` (or automatically during `bytx run`)
   produces a summary per video via map-reduce over the transcript, in the
   video's own language by default (`summarize.language`).

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

# Queue from a text file (one URL per line, '#' comments allowed)
uv run bytx add urls.txt --run

# Facebook videos work the same way, and can share a file with YouTube ones
uv run bytx add "https://www.facebook.com/watch/?v=123456789012345" --run
uv run bytx add "https://www.facebook.com/reel/123456789012345"
uv run bytx add "https://fb.watch/abcDEF12/"

# Check on things
uv run bytx status
uv run bytx list --status transcribed
uv run bytx list --lang en
uv run bytx list --platform facebook

# Read / export a transcript
uv run bytx show <video_id>
uv run bytx export <video_id> --format srt --output out.srt

# Search
uv run bytx search "বাংলাদেশ"
uv run bytx search "বাংলাদেশের ইতিহাস" --semantic
uv run bytx search "student visa" --semantic

# Ask questions across your library (or one video with --video)
uv run bytx ask "এই ভিডিওতে কী বলা হয়েছে?"
uv run bytx ask "What documents do I need for the visa?"

# Retry anything that failed
uv run bytx retry && uv run bytx run
```

The pipeline is resumable: if it's interrupted (Ctrl+C, crash, out of
disk), just run `bytx run` again — already-finished videos are skipped and
nothing is redone.

## Configuration

All settings live in `config.toml` at the project root: which ASR model
serves each language, VAD/decoding parameters, which Ollama models to use
for embeddings and chat, chunk sizes for retrieval, and yt-dlp options
(cookies, sleep intervals). See the comments in that file.

### Facebook

Facebook support goes through yt-dlp, so nothing extra needs installing.

- **What works:** single videos and reels: `facebook.com/watch/?v=…`,
  `/<page>/videos/<id>`, `/reel/<id>`, `video.php`, and the `m.`/`web.` hosts.
  Other Facebook URLs that yt-dlp understands (`permalink.php`, posts, ...) are
  looked up once to find the video's id.
- **Short and share links** (`fb.watch/…`, `facebook.com/share/v/…`) have no
  id in them. yt-dlp has no extractor of its own for them, so they only work
  if it can follow the redirect to the real video, which Facebook sometimes
  blocks behind a login. One that can't be resolved is skipped with a warning;
  add the full video URL instead.
- **Pages and playlists:** yt-dlp currently has no Facebook page/playlist
  extractor, so `bytx add` of a page's `/videos` tab queues nothing. Add the
  individual video URLs (a text file works well).
- **Private, friends-only and group videos** need you to be logged in. Set
  `cookies_from_browser` under `[fetch]` in `config.toml` (for example
  `"firefox"`), and yt-dlp will borrow that browser's Facebook session.
- **Transcription:** Facebook offers no uploader subtitles, so these always
  use ASR. A video with no title is named by the first line of its caption.
- **Ids:** YouTube videos keep their bare 11-character id. Facebook videos are
  stored as `fb_<number>` (e.g. `fb_123456789012345`), which is what `bytx list`
  shows and what `show`, `export`, `summarize` and `ask --video` take. The
  ledger and `data/videos/` use the same ids. Search results and citations link
  to the right site with a timestamp, and each video's platform is in the
  `videos.platform` column and `info.json`.
- **Other sites** are skipped with a warning rather than queued.

### Languages

Every video has a language, `bn` (Bangla) or `en` (English), stored in the
database (`videos.language`), the ledger, and the JSON exports. It picks the
ASR model and which uploader subtitles are used. You choose it: `bytx add
--lang en` sets it for everything that `add` queues, and without `--lang` it
is `asr.default_language` (`bn`).

Nothing guesses a video's language unless you ask (`--lang auto`), on purpose.
Measured on this project's own library, both obvious signals are wrong for
Bangla videos that mix in English words (very common here):

- YouTube's language field is the channel default and reads `en-US` on
  about half of the Bangla videos sampled.
- Whisper's language ID gave Bangla a probability of at most 0.03 on all 25
  videos tried. Bangla and Banglish were called Hindi or English (0.7-0.99),
  which is indistinguishable from Bangladeshi-accent English.

Trusting either would silently send Bangla videos to the English model.

**`--lang auto`** queues videos with no language, and `bytx run` decides each
one first, using a different signal: it transcribes a sample of the audio
(ten clips spread across the video, about 90 s of speech) with the Bangla
model and counts how many of the words are common Bangla grammar words
(আছে, করে, এবং, না, ...). Bangla speech, even full of English words, comes
out at 12.9%-29.3% (27 videos measured). English speech comes out as
Bengali-script transliteration with few of them: 1.5%-8.5% (5 videos). The
video is English when the share is below `asr.auto_english_below` (10%, in the
gap between the two), Bangla otherwise. The audio downloaded for the sample is
reused for the transcription, so nothing is fetched twice. A sample too short
to judge (under 40 words) falls back to `asr.default_language` and logs a
warning.

Limits: it only chooses between Bangla and English (a Hindi video would be
called English), it judges the video as a whole rather than per sentence, and
those thresholds come from one library of study-abroad videos. Check
`bytx list` after an `auto` run, and use `--lang` when you know.

Uploader subtitles are only used when they are in the video's language, so a
Bangla video that carries just English subtitles is still transcribed from
its audio.

`[asr.language_models]` maps each language to a model:

```toml
[asr]
default_language = "bn"

[asr.language_models]
bn = "tugstugi-medium-bn"   # Bangla fine-tune, pinned to bn
en = "large-v3-turbo"       # multilingual (language = "" in its model entry)
```

Only one ASR model is held in memory at a time. `bytx run` groups the queue
by language so each model is loaded once (detection uses the Bangla model, so
`auto` videos add no extra loads). Model entries can set their own
`max_clip_seconds`: English needs far fewer tokens per second than Bangla, so
`large-v3-turbo` uses 15 s clips against the global 12 s. A config with no
`[asr.language_models]` (from before English support) keeps working as a
Bangla-only setup.

Existing libraries are upgraded automatically the first time the database is
opened: videos transcribed before English support are marked `bn`.

### How audio is cut into segments

Whisper stops generating after **224 tokens per 30 s window**, and Bengali
needs about 0.7 tokens per character (English about 0.3, which is why English
can use longer clips). A fast or dense 30 s window therefore
overflows: the text is cut off mid-sentence (sometimes mid-word) and the rest
of that window is silently lost. Early transcripts were affected (94% of
segments sat exactly at the cap; on one fast speaker ~40% of the words were
missing).

To avoid this, `Transcriber.transcribe()` does its own segmentation instead of
handing the whole file to faster-whisper:

1. Silero VAD finds speech, and adjacent speech is merged into clips of at most
   `asr.max_clip_seconds` (12 s; a model entry can override it) at natural pauses.
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

- Numbers are written as spoken words ("দুই হাজার সাতাইশ"), not digits (Bangla).
- English words inside Bangla speech are transliterated into Bengali script
  and are often misheard ("এস অপি" for SOP, "স্টোর" for story).
- No punctuation in Bangla; segment breaks are speech pauses, not sentences.
- Music/outro sections that slip past VAD can produce nonsense text.
- A video is transcribed in one language. A video that switches between
  Bangla and English is decoded entirely in whichever language it was
  assigned to, so the other language comes out transliterated or garbled.
- English subtitles/cues are clip-sized (up to 15 s) rather than
  sentence-sized, because timestamps are taken from the clip boundaries.

### Choosing an ASR model

Two models are pre-configured:

- `tugstugi-medium-bn` (Bangla) — a Bangla fine-tune of Whisper-medium.
  Best accuracy on Bangla speech; pinned to Bangla.
- `large-v3-turbo` (English) — multilingual, so it serves English and can
  also be tried on Bangla.

`bytx setup` downloads both (`large-v3-turbo` is a ~1.6 GB pre-converted
download; the Bangla model is converted locally). Compare models on your own
content before changing which one serves a language:

```
uv run bytx bench "<url>" --clip 01:00-03:00 --models tugstugi-medium-bn,large-v3-turbo --lang bn
uv run bytx bench "<english url>" --clip 01:00-03:00 --models large-v3-turbo --lang en
```

`--lang` defaults to each model's own language (English for a multilingual
model). A model pinned to one language reports an error if asked for another.
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
  langs.py        supported languages: codes, names, normalization
  fetch.py        yt-dlp: URL expansion, metadata, audio download, subtitles
  transcribe.py   faster-whisper wrapper (VAD, language detection, per-language model pool, fallback chain, anti-hallucination)
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
yt-transcribed-files/   git-tracked <date>/"<channel> - <title>" transcript folders
```

### Transcribed-video ledger and human-facing output

`data/` (SQLite DB + per-id export folders) is gitignored, so on its own it
can't tell a fresh clone, or a wiped `data/`, that a video has already been
transcribed. Two things fix that:

- **`TRANSCRIBED_LEDGER.md`** — a plain, append-only markdown record (video
  id, channel, title, url, language, date) written after every successful
  transcription. `bytx add` checks it alongside the DB, so a known video id
  is never silently re-queued.
- **`yt-transcribed-files/<YYYY-MM-DD>/<channel> - <title>/transcript.txt`** —
  a human-browsable, git-tracked copy of the plain-text transcript, filed
  under the date it was transcribed and separate from the DB-linked
  `data/videos/<id>/` export. Re-running a video keeps its original date
  folder.

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
