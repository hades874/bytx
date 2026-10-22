# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`bytx` is a local, free/open-source CLI that transcribes Bangla and English YouTube/Facebook videos (faster-whisper) into a searchable, askable library (SQLite FTS5 + Ollama). README.md is detailed and accurate; read it for user-facing behavior, config semantics and the measured rationale behind design choices.

Distribution roadmap and settled decisions: `docs/PRD-distribution.md`. Read it before changing install, config/data paths, models, dependencies or the UI. Task breakdown (pick the next unchecked task there): `docs/PLAN-distribution.md`.

## Commands

Python 3.12, managed with `uv`.

```
uv sync --extra gpu              # dev env with CUDA libs (plain `uv sync` without NVIDIA, and it removes them); --extra convert adds torch
uv run pytest                    # offline suite, no GPU/network (testpaths = tests)
uv run pytest tests/test_store.py::test_name   # single test
uv run ruff check .              # lint (line length 110; tests exempt from E501)
uv tool install --editable ".[gpu]"   # an installed `bytx` on PATH that runs this working copy
```

Daily use goes through the installed `bytx`, like any user's: `bytx setup` downloads the ASR models into the Hugging Face cache and pulls Ollama models if Ollama runs; `bytx doctor` checks the data folder, Deno, models, GPU and Ollama.

Entry point is `bytx = "bytx:main"` (`cli.py`, Typer). Settings: the packaged `src/bytx/defaults.toml`, with the user's `config.toml` from the data folder deep-merged over it (`config.py`). The data folder is `$BYTX_HOME`, else `~/bytx`; `--config <file>` makes that file's folder the data folder. ASR models are found and downloaded only through `models.py` (a local CTranslate2 folder as `source`, or the HF cache).

## Transcribing videos

When asked to transcribe new videos (e.g. those in `urls.txt`), follow `docs/SOP-video-transcription.md`. The two rules that matter most: always pass an explicit `--lang` to `bytx add` (never rely on `asr.default_language`; an English video queued as `bn` produces Bangla-script gibberish that no quality flag catches), and verify the transcript text yourself before reporting success (`failed=0` does not mean correct). To redo a video, use `bytx remove <id>` (dry run; `--yes` deletes), and ask before deleting.

## Architecture

**Pipeline** (`pipeline.py`): each video moves through a status ladder `queued -> transcribed -> indexed -> summarized` (or `failed`). Every stage re-queries its input status, so `bytx run` is resumable and idempotent. Stage 1 (faster-whisper on GPU) unloads its model before stages 2/3 (embeddings, summaries) use Ollama, so the two never share VRAM.

**Language is an explicit, recorded decision.** Each video has `bn` or `en` (`videos.language`), set by `bytx add --lang` (default `asr.default_language`) and used to pick the ASR model (`[asr.language_models]`) and which uploader subtitles are acceptable. Never introduce Whisper per-chunk language ID or YouTube's language metadata: both mislabel Bangla-with-English-words as English/Hindi. `--lang auto` queues with no language and `langdetect.py` decides at run time by counting Bangla function words in a sample transcribed with the Bangla model. `TranscriberPool` holds one model at a time and `run` groups the queue by language so each model loads once.

**Custom segmentation in `transcribe.py`**: Whisper truncates at 224 tokens per 30 s window (Bengali ~0.7 tokens/char), so `Transcriber.transcribe()` does its own Silero-VAD clipping (`max_clip_seconds`, per-model override), decodes each clip separately, and re-splits any clip near the token cap. Segments are clip-sized with real start/end times because the Bangla model doesn't emit timestamp tokens (`without_timestamps = true`). Do not re-enable `repetition_penalty` / `no_repeat_ngram_size` (forces a much slower decode path); `collapse_repeated_ngrams()` handles repeats instead.

**Raw ASR text is never modified.** `glossary.py` produces a separate `transcript.clean.txt` (find/replace of known mis-hearings) and primes the decoder for languages in `asr.glossary_prompt_languages`. `quality.py` computes flags (words/min, gaps, low `avg_logprob`/`no_speech_prob`) from stored segments; `compare.py` cross-checks against YouTube auto-captions. `pipeline.reexport` rebuilds all derived files from the DB.

**Storage is three-layered, all inside the data folder (never the repo):**
- `data/library.db` (`store.py`): SQLite with a trigram FTS5 index, embedding chunks and summaries, plus schema migrations via `_migrate` (existing rows upgraded on open). `data/videos/<id>/` holds DB-linked exports.
- `yt-transcribed-files/<date>/<channel> - <title>/` (`export.py`): human-readable copy, with a hidden `.video_id` file tying folder to video, and a generated `INDEX.md`.
- `TRANSCRIBED_LEDGER.md` (`ledger.py`): append-only record of transcribed ids, checked by `bytx add` so videos are never re-transcribed after `data/` is wiped. `bytx remove` (`pipeline.remove_video`) clears a video from all three.

**Ollama is optional.** `pipeline.run` raises `models.ModelsMissing` before touching any video when ASR models aren't downloaded (videos stay `queued`), and returns early with `RunSummary.stopped` when Ollama or its models are missing (videos stay `transcribed`). Neither case may mark videos `failed`.

**Platforms** (`platforms.py`, `fetch.py`): YouTube ids are stored bare (11 chars); Facebook ids are `fb_<number>`. Everything downstream (paths, ledger, CLI args) uses these ids. Facebook has no uploader subtitles and no playlist extractor in yt-dlp.

**Retrieval/LLM**: `rag.py` (time-windowed chunks, `bge-m3` embeddings, `ask` with timestamped citations), `summarize.py` (map-reduce), `llm.py` (minimal Ollama HTTP client). `cuda_dll.py` fixes Windows CUDA/cuDNN DLL discovery (both `add_dll_directory` and `PATH` are required for CTranslate2); it should not need touching.

## Testing notes

`tests/conftest.py` puts `src/` on `sys.path` and configures UTF-8 stdio (Bangla output on Windows consoles); `textio.configure_utf8_io` does the same for the CLI. An autouse fixture there points `BYTX_HOME` at a temp folder, so no test touches the real data folder. Tests are fully offline, so mock yt-dlp, faster-whisper, Hugging Face downloads and Ollama rather than calling them.
