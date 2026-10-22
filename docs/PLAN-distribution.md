# Plan: building the distribution PRD

| | |
|---|---|
| Implements | [`docs/PRD-distribution.md`](PRD-distribution.md) (approved 2026-10-22) |
| Branch | `distribution` (from `master`) |
| Audience | The maintainer, contributors, and coding agents working in this repo |

The PRD says *what* and *why*. This file splits it into tasks that are each about one commit or PR, and settles the design questions the PRD left open. When this file and the PRD disagree on a requirement, the PRD wins. When they disagree on how to build it, this file wins.

## How to work from this file

1. Pick the first unchecked task whose **Needs** are all checked (see [Order](#order)).
2. Read the files it lists before changing them. The line numbers were right on 2026-10-22.
3. Finish **Done when** and **Tests**, then run `uv run pytest` and `uv run ruff check .`. Both must pass.
4. Tick the task's box in the same commit as the work.
5. Tasks marked **Maintainer** need Hasan's accounts, hardware or judgment. Prepare everything around them, then stop and ask.

## Settled design

These are the decisions the PRD left open. Don't re-open them without the maintainer.

- **ASR models live in the Hugging Face cache, not in `~/bytx`.** `HF_HOME` moves the cache. `paths.models_dir` is deleted. A new module, `src/bytx/models.py`, is the only code that knows where models are:
  - `local_path(model_cfg) -> Path`. If `source` is an existing folder it is used as it is (a converted or maintainer model). Otherwise it calls `snapshot_download(source, local_files_only=True)` and checks that `model.bin` exists. Raises `ModelsMissing` with "run `bytx setup`".
  - `download(model_cfg, force=False) -> Path`. This is the `already_ct2` branch of `scripts/convert_model.py:convert_or_download`, moved into the package. A Hugging Face source that is not CTranslate2 raises a clear error: "convert it with scripts/convert_model.py and set `source` to the output folder".
  - `missing(cfg) -> list[str]`: the configured model keys that are not downloaded.
- **Data folder layout.** `Config.root` is the data folder, so everything that resolves against it (ledger, glossary, paths) moves with it:

  ```
  ~/bytx/                      (or $BYTX_HOME)
    config.toml                overrides only; optional
    data/library.db, data/videos/, data/audio/
    yt-transcribed-files/
    TRANSCRIBED_LEDGER.md
    glossary.txt
  ```

  The `data/` subfolder stays as it is, so the maintainer migration is a plain folder copy.
- **Defaults ship in the package.** `config.toml` moves to `src/bytx/defaults.toml` and is read with `importlib.resources`. The user's file is deep-merged over it: tables merge, scalars and lists replace. `load_config(path=None)` keeps its interface. Resolution order is `--config` (root = its parent, and the file must exist), then `BYTX_HOME`, then `~/bytx`.
- **`pipeline.run` stops in two ways.**
  - Models missing: it raises `models.ModelsMissing` before any stage runs. Videos stay `queued` and the CLI exits 1.
  - Ollama unreachable, or its embed or chat model not pulled: it returns early with `RunSummary.stopped = "<hint>"`. Videos stay `transcribed` and the CLI exits 0.

  Both checks live inside `pipeline.run`, so `bytx serve` gets them without extra code.
- **`remove` is one function:** `pipeline.remove_video(conn, cfg, video_id, apply: bool) -> Removal`. The CLI and the web Video page both call it. It rejects any id that doesn't match `[A-Za-z0-9_-]+`. Without that guard, `scripts/redo_video.py ..` deletes `data/`, `library.db` included.
- **CPU threads come from `psutil.cpu_count(logical=False)`.** The standard library only counts logical cores, and halving that would drop a 4-core CPU without hyperthreading below today's 4 threads.
- **`doctor` must work in a `uv tool` install.** There the `yt-dlp` executable is not on PATH. Read the version from `yt_dlp.version.__version__` instead.
- **Web security (R2).**
  - `TrustedHostMiddleware(["127.0.0.1", "localhost"])` blocks DNS rebinding.
  - Every POST is checked for a same-origin `Origin` header, so another website can't trigger Remove or Add.
  - Jinja autoescape stays on, because titles and transcripts are untrusted. Don't use `|safe` on them.
  - `HOST = "127.0.0.1"` is defined in one place. There is no `--host` flag.

## Maintainer actions (blocking)

| Action | Blocks |
|---|---|
| Create a Hugging Face account and the `bytx` organisation | T8 |
| Upload the Bangla model with your own token (commands prepared in T8) | T8 |
| Create the PyPI pending trusted publisher: project `bytx`, owner `hades874`, repo `bytx`, workflow `ci.yml`, environment `pypi` | T10 |
| Run the real-download check with ffmpeg off PATH, and the CPU benchmark on the Ryzen 5 5600 | T5d |
| Run the fresh-VM checklists (Windows, Ubuntu, macOS; later Windows 10/11 for `setup.exe`) | T10, U13 |
| Take the SmartScreen screenshot | U12 |
| Migrate your own library, and approve deleting the repo copies | T11 |
| Run the tester round with 2–3 non-technical users | U13 |

---

## Phase 0

### S0. Spike: downloads with ffmpeg and Deno off PATH
- [x] Done (2026-10-22). yt-dlp finds the pip Deno with nothing on PATH, and a YouTube download decodes without ffmpeg, so T5c needs no `js_runtimes` option. Facebook is still unchecked (no Facebook video in the library); T5d covers it. Details in PRD §9. · **Needs:** nothing · **PRD:** R1.5, §9 open question 3 · **Blocks:** T5c and the ffmpeg drop

**Do:** Open a fresh shell *without* `env.ps1` and create a scratch venv outside the repo with `uv pip install "yt-dlp[default,deno]" faster-whisper`. Check that `where ffmpeg` and `where deno` find nothing. Download `bestaudio/best` for one YouTube URL and one Facebook URL using the same options as `fetch.download_audio`, then decode each file with `faster_whisper.decode_audio`.

**Done when:** PRD §9 records the answers to two questions. Does yt-dlp find the pip-installed Deno on its own? Does PyAV decode both downloads with no system ffmpeg? If yt-dlp does not find Deno, T5c also passes the pip Deno binary to yt-dlp through `js_runtimes` in `fetch._base_ydl_opts`.

## Release 1 (`0.2.0`): code

### T1a. Packaged defaults + deep merge
- [x] Done · **Needs:** nothing · **PRD:** R1.2

**Files:** `config.toml` → `src/bytx/defaults.toml` (`git mv`), `src/bytx/config.py`, `tests/test_config.py`, other tests that write partial configs.

**Do:**
- Rewrite the header of `defaults.toml`: these are the packaged defaults, don't edit them, put overrides in `~/bytx/config.toml`, and relative paths resolve against the data folder.
- In `load_config`, read the defaults with `importlib.resources.files("bytx") / "defaults.toml"`, read the user's file if there is one, and merge with a small recursive `_merge(base, over)`.
- Keep the root logic as it is for now (T1b changes it).

**Done when:** a missing or empty user file runs on the defaults, and a user file holding only `[asr]\ndevice = "cpu"` changes only `asr.device`.

**Tests:**
- Merge semantics: nested tables merge; scalars and lists replace.
- The empty user file and the device-only file above.
- Point `SHIPPED_CONFIG` at the packaged file.
- Fix any test that assumed `[asr.language_models]` is empty when the file leaves it out; merged defaults now fill it.

### T1b. Data folder resolution (`~/bytx`, `BYTX_HOME`)
- [x] Done · **Needs:** T1a · **PRD:** R1.1

**Files:** `src/bytx/config.py`, `src/bytx/cli.py` (`_config_option` help text), `tests/conftest.py`, `tests/test_config.py`.

**Do:**
- Delete `_find_project_root` (`config.py:335`).
- Add `data_home()`, which returns `Path(os.environ["BYTX_HOME"])` if that is set and `Path.home() / "bytx"` otherwise.
- Without `--config`, root is `data_home()` and the user file is `root / "config.toml"`, which may be missing.
- Remove the `models_dir` line from `ensure_dirs`. It creates the root through `parents=True`.
- Add an **autouse** fixture to `tests/conftest.py` that sets `BYTX_HOME` to a `tmp_path`, so no test touches the real home folder.

**Done when:** `bytx status` works from any folder. The first command on a machine with no `~/bytx` creates it with no error or prompt. `BYTX_HOME` moves everything, and `--config` still wins over it.

**Tests:**
- Each of the three resolution paths. For `~/bytx`, monkeypatch `Path.home`.
- `--config` wins over `BYTX_HOME`.
- `ledger.ledger_path(cfg)` lands in the data folder.
- A CLI `status` run against an empty `BYTX_HOME` creates the folder.

### T2a. `models.py`; setup downloads only
- [x] Done · **Needs:** T1b · **PRD:** R1.3

**Files:** new `src/bytx/models.py`, `src/bytx/cli.py` (`setup`), `scripts/convert_model.py`, new `tests/test_models.py`.

**Do:**
- Implement `local_path`, `download`, `missing` and `ModelsMissing` as described in [Settled design](#settled-design).
- `setup` loops over `cfg.asr.required_model_keys()` and calls `models.download` in-process. Delete the subprocess call and the `scripts_dir` lookup (`cli.py:167-188`).
- Reduce `scripts/convert_model.py` to `convert_model.py <hf-source> <out-dir> [--quantization float16] [--force]` with no config loading. It stays a maintainer tool that needs the `convert` extra.

**Done when:** `bytx setup` completes on an install without the `convert` extra. Until T8 lands, the default Bangla model is not CTranslate2, so setup stops with the "convert it" error. The maintainer works around that with a local-folder `source` in their user config.

**Tests:** mock `huggingface_hub.snapshot_download`. Cover:
- a local folder is returned as-is;
- a cached repo returns its path;
- an uncached repo raises `ModelsMissing`;
- a non-CTranslate2 Hugging Face source makes `download` raise;
- `missing` lists only the absent keys.

### T2b. Read models through `models.local_path`
- [x] Done · **Needs:** T2a · **PRD:** R1.3

**Files:** `src/bytx/transcribe.py` (`TranscriberPool._get` at line 472, the `Transcriber._load` message at line 236), `src/bytx/bench.py` (`_bench_one_model`, line 90), `src/bytx/cli.py` (`doctor` model loop), `src/bytx/config.py` (`PathsConfig`). Test fixtures that build `PathsConfig`: `test_pipeline.py`, `test_rag.py`, `test_summarize.py`, `test_fetch.py`, `test_transcribe.py`.

**Do:**
- Replace every `cfg.paths.models_dir / key` with `models.local_path(cfg.asr.models[key])`.
- Delete `PathsConfig.models_dir` and its parsing.
- `doctor` and `bench` print the `ModelsMissing` message instead of their own.

**Done when:** `grep -r models_dir src tests` finds nothing.

**Tests:** the existing pool, bench and doctor tests, now monkeypatching `models.local_path`.

### T2c. `run` before `setup` keeps videos queued
- [x] Done · **Needs:** T2b · **PRD:** R1.3

**Files:** `src/bytx/pipeline.py` (`run`, line 267), `src/bytx/cli.py` (`_do_run`), `tests/test_pipeline.py`, `tests/test_cli.py`.

**Do:**
- In `pipeline.run`, when the queue is not empty, call `models.missing(cfg)`. If anything is missing, raise `ModelsMissing` naming the keys.
- `_do_run` catches it, prints the message ("run `bytx setup`") and exits 1.

**Done when:** `bytx run` before `bytx setup` stops with that message, and every video stays `queued`.

**Tests:** with models missing, `run` raises, every video stays `queued` and none is `failed`. The CLI exits 1 with "bytx setup" in its output.

### T3a. Ollama gate in `run`
- [x] Done · **Needs:** T1b · **PRD:** R1.4

**Files:** `src/bytx/pipeline.py` (`RunSummary`, `run` lines 303-320), `src/bytx/cli.py` (`_do_run`), `tests/test_pipeline.py`.

**Do:**
- Add `stopped: str | None = None` to `RunSummary`.
- If there are videos to index or summarize, open one `OllamaClient` before the index stage and check `is_reachable()`, `has_model(embed_model)`, and `has_model(chat_model)` when summarizing.
- If any check fails, set `stopped` to a one-line hint and return. Use "Ollama isn't running: transcripts are saved; start Ollama and run `bytx run` again to index and summarize", or "run `bytx setup`" for missing models.
- Reuse that one client for both stages. `_do_run` prints `stopped` in yellow.

**Done when:** with Ollama stopped, `bytx add <url> --lang bn --run` leaves the video `transcribed` with `failed=0`. Starting Ollama and running `bytx run` moves it through `indexed` and `summarized`.

**Tests:**
- Unreachable leaves `transcribed`, `failed == 0` and `stopped` set.
- Reachable but no embed model sets the "run `bytx setup`" hint.
- Reachable with models reaches `summarized`.

### T3b. Ollama-optional commands, setup and doctor
- [x] Done · **Needs:** T3a · **PRD:** R1.4

**Files:** `src/bytx/cli.py` (`ask`, `search`, `summarize`, `setup`, `doctor`), `tests/test_cli.py`.

**Do:**
- Add one helper, `_ollama_client(cfg) -> OllamaClient`. If Ollama is unreachable it prints "This needs Ollama (optional). Install it from https://ollama.com, then run `bytx setup`." and exits 1.
- Use it in `ask`, `search --semantic` and `summarize`.
- `setup`: if Ollama is unreachable, print that the Q&A and summary models were skipped and how to add them later. Don't fail.
- `doctor`: show an unreachable Ollama as a yellow `WARN (optional)` that doesn't set `ok = False`. Do the same for missing Ollama models.

**Done when:** with Ollama absent, `show`, `export`, keyword `search` and `quality` work. `ask`, `search --semantic` and `summarize` print the install hint. `setup` and `doctor` still succeed.

**Tests:** cover each of those with `OllamaClient.is_reachable` monkeypatched to `False`.

### T4. `bytx remove <id>`
- [x] Done · **Needs:** T1b · **PRD:** R1.6

**Files:** `src/bytx/ledger.py`, `src/bytx/pipeline.py`, `src/bytx/cli.py`; `scripts/redo_video.py` (delete); `tests/test_ledger.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.

**Do:**
- Move `_ENTRY_RE` and `_strip_ledger_entry` from `redo_video.py` into `ledger.remove_entry(path, video_id) -> bool`.
- Add a `Removal` dataclass to `pipeline.py` (video, db_row, data_dir, shared_folder, ledger_entry, plus an `empty` property) and `remove_video(conn, cfg, video_id, apply)`.
- Port the script's behaviour exactly. The DB row goes (segments, chunks and summary cascade), then `videos/<id>/`, then the shared folder and its date folder if that is now empty. `export.write_index` rebuilds the index, then the ledger entry goes.
- Reject ids that don't match `[A-Za-z0-9_-]+` before touching any path.
- CLI: `bytx remove <id> [--yes]` prints what was found. If nothing was found it says so and exits 1. Without `--yes` it is a dry run that exits 0.
- Delete the script.

**Done when:** the dry run lists exactly what would be removed and changes nothing. After `--yes`, `bytx add` accepts the same URL again. An unknown id exits non-zero with a clear message.

**Tests:** the dry run leaves the DB, folders and ledger unchanged. `--yes` removes all four and rebuilds `INDEX.md`. `add_targets` re-queues the id afterwards. An unknown id exits 1. `..` and `a/b` are rejected and nothing is deleted.

### T5a. Dependencies + `cuda_dll` fix
- [x] Done · **Needs:** nothing (do it after T4 to keep diffs apart) · **PRD:** R1.5

**Files:** `pyproject.toml`, `uv.lock`, `src/bytx/cuda_dll.py`, new test in `tests/test_transcribe.py` or `tests/test_cuda_dll.py`.

**Do:**
- Move `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` into a `gpu` extra, each with `; sys_platform == 'win32' or sys_platform == 'linux'`.
- Change `yt-dlp[default]` to `yt-dlp[default,deno]` and add `psutil`. Run `uv lock`.
- In `cuda_dll.py:41`, wrap `find_spec(pkg)` in `try/except ModuleNotFoundError: continue`. `find_spec("nvidia.cublas")` raises when the parent `nvidia` package is absent.

**Done when:** `uv sync` without extras installs no `nvidia-*` package, and `ensure_cuda_dll_paths()` returns `[]` on that install.

**Tests:** with `sys.platform` patched to `win32` and `find_spec` raising `ModuleNotFoundError`, the function returns `[]`.

### T5b. CPU threads = physical cores
- [x] Done · **Needs:** T5a · **PRD:** R1.5

**Files:** `src/bytx/transcribe.py` (`Transcriber._load`, line 253), `tests/test_transcribe.py`.

**Do:** when `device == "cpu"`, pass `cpu_threads=psutil.cpu_count(logical=False) or os.cpu_count() or 4`. Leave the call unchanged on CUDA.

**Tests:** monkeypatch `WhisperModel` to record its kwargs. CPU gets the physical-core count; CUDA gets no `cpu_threads`.

### T5c. `doctor` and `bench` for an installed `bytx`
- [x] Done · **Needs:** S0, T3b, T5a · **PRD:** R1.5

**Files:** `src/bytx/cli.py` (`doctor`), `src/bytx/bench.py`, `src/bytx/fetch.py` (only if S0 says so), `tests/test_cli.py`.

**Do:**
- Drop the ffmpeg check.
- Check Deno with the same lookup yt-dlp uses (per S0); if T5c sets `js_runtimes`, check that path.
- Get the yt-dlp version from `yt_dlp.version.__version__`.
- Print the data folder and whether `config.toml` exists there ("defaults in use").
- GPU hint: if the `nvidia` package is not importable, run `nvidia-smi -L` (if it is on PATH). If it lists a GPU, suggest `uv tool install "bytx[gpu]"`.
- Show the "fell back from cuda" note only when the gpu extra is installed.
- `bench`: say in its docstring and `--help` that `--clip` needs ffmpeg, and turn a `FileNotFoundError` from the ffmpeg subprocess into a friendly message.

**Done when:** on a plain install with no system ffmpeg or Deno, `doctor` reports no failures. On the RTX 4070 with `bytx[gpu]`, `doctor` shows `cuda/float16` with no fallback note.

**Tests:** replace `test_doctor_says_deno_is_only_needed_for_youtube`. Add: no ffmpeg check, the GPU hint (mock `shutil.which` and `subprocess.run`), and the data folder line.

### T5d. Manual gates for R1.5 (Maintainer)
- [ ] Done · **Needs:** T5c · **PRD:** R1.5

**Do:**
- With ffmpeg off PATH and no system Deno, `bytx add <youtube-url> --lang <x> --run` and the same for one Facebook URL must both succeed. Read each transcript. **This check gates dropping ffmpeg.**
- Re-measure the CPU real-time factor on the Ryzen 5 5600 with `bytx bench` on the same 180 s clip, and compare with the 2.25 baseline.

**Done when:** both results are written into PRD §11.

### T6. CI and release workflow
- [ ] Done · **Needs:** T5a · **PRD:** R1.7, R1.5 (macOS install)

**Files:** new `.github/workflows/ci.yml`, `pyproject.toml` (metadata).

**Do:** one workflow, triggered on push, pull request and `v*` tags, with three jobs:
- `test`, matrix `windows-latest` and `ubuntu-latest`: `astral-sh/setup-uv` (pinned), `uv sync --locked`, `uv run ruff check .`, `uv run pytest`.
- `smoke`, matrix `windows-latest`, `ubuntu-latest` and `macos-latest`: `uv build`, `uv tool install` the built wheel, then from `$RUNNER_TEMP` with a temporary `BYTX_HOME` run `bytx --help` and `bytx status`. This proves "works from any directory" and "installs on macOS" on every push.
- `publish`, only on `v*` tags and `needs: [test, smoke]`: `environment: pypi`, `permissions: id-token: write`. Check the tag matches `uv version --short`, then `uv build` and `uv publish` (trusted publishing, no token).

Add `[project.urls]`, `classifiers` and `keywords` to pyproject.

**Done when:** CI is green on the branch. A failing test makes `publish` skip, because of `needs`.

### T7. Docs and repo cleanup
- [ ] Done · **Needs:** T1a–T5c, T6 · **PRD:** R1.8

**Files:** `README.md`, `CLAUDE.md`, `docs/SOP-video-transcription.md`, `.gitignore`, `pyproject.toml` (the `testpaths` comment).

**Do:**
- README covers: install (`uv tool install bytx` and `uv tool install "bytx[gpu]"`); then `bytx setup`, `bytx doctor`, `bytx add <url> --lang bn --run`; the data folder and `BYTX_HOME`; Ollama as optional; CPU speed (about 2.25× the audio's length on a 6-core desktop, slower on laptops); macOS CPU-only; the Linux GPU `LD_LIBRARY_PATH` note marked community-tested; the glossary format (moved from the header of `glossary.txt`); and a responsible-use note (respect copyright and platform terms). Also say that `bench --clip` needs ffmpeg.
- CLAUDE.md and the SOP target the installed `bytx`: no `env.ps1`, no `uv run` for daily use, and `bytx remove` instead of the redo script.
- Run `git rm --cached env.ps1 urls.txt glossary.txt` and add those three files to `.gitignore`. The local copies stay until T11.

**Done when:** a README-only walkthrough reaches a transcript. That is checked for real in T10.

## Release 1: maintainer-gated

### T8. Publish the Bangla model on Hugging Face
- [ ] Done · **Needs:** T2a, plus the maintainer's Hugging Face organisation · **PRD:** R1.3

**Do (engineering):**
- Write `docs/hf/tugstugi-medium-bn-ct2.md`: model-card front matter (`license: apache-2.0`, `language: bn`, `library_name: ctranslate2`, tags `faster-whisper` and `whisper`, `base_model: bengaliAI/tugstugi_bengaliai-asr_whisper-medium`), credit to bengaliAI, float16 CTranslate2, the conversion command, and how bytx uses it.
- Prepare the upload commands. The local `models/tugstugi-medium-bn/` (model.bin, config.json, tokenizer.json, preprocessor_config.json, vocabulary.json) is already float16 CTranslate2, so it uploads as it is:

  ```
  uv run hf auth login
  uv run hf upload bytx/tugstugi-medium-bn-ct2 models/tugstugi-medium-bn . --repo-type model
  uv run hf upload bytx/tugstugi-medium-bn-ct2 docs/hf/tugstugi-medium-bn-ct2.md README.md
  ```

**Do (Maintainer):** run the commands with your own token.

**Then:** in `defaults.toml`, set `[asr.models.tugstugi-medium-bn]` to `source = "bytx/tugstugi-medium-bn-ct2"` and `already_ct2 = true`. Add a test that every model in the shipped defaults is `already_ct2`, so setup never needs torch.

**Done when:** `bytx setup` on a fresh install downloads both models with no `convert` extra.

### T9. Release 1 P1 items
- [ ] Done · **Needs:** T8 · **PRD:** R1 P1

**Do:**
- Before downloading, `setup` prints each model's size, using `HfApi().model_info(source, files_metadata=True)` and summing the sibling sizes. Skip models already present. Mock this in tests.
- Write `docs/README.bn.md`, a Bangla quick-start for technical users, linked from the README.

### T10. Release `0.2.0` (Maintainer + engineering)
- [ ] Done · **Needs:** T6, T7, T8, plus the PyPI pending publisher

**Do:**
- Run the fresh-VM checklist on Windows, Ubuntu and macOS (CPU), following only the README, to a transcript. Record the measured first-time download sizes (plain, `[gpu]`, with Ollama) in the README and PRD §9.
- Bump `version` to `0.2.0`, then tag and push `v0.2.0`.

**Done when:** `uv tool install bytx` installs 0.2.0 from PyPI.

### T11. Maintainer migration (PRD §7)
- [ ] Done · **Needs:** T10 (or T8 for an editable install before the release)

**Do:**
1. `uv tool install --editable ".[gpu]"` from the clone.
2. Set permanent user environment variables: `BYTX_HOME` (a folder on `D:\`), `HF_HOME`, `OLLAMA_MODELS`, and `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` if wanted.
3. Look at the target, then copy `data/`, `yt-transcribed-files/`, `TRANSCRIBED_LEDGER.md`, `glossary.txt` and `urls.txt` into `BYTX_HOME`.
4. Run `bytx setup`. It downloads from Hugging Face rather than reusing `models/`, which also checks T8 end to end.
5. Verify: `bytx status` shows 45 videos (39 bn, 6 en, all `summarized`), `bytx search` returns hits, and `bytx ask` cites timestamps.
6. Only after verifying, and **after asking**, delete the old copies from the repo folder (`data/`, `models/`, `yt-transcribed-files/`, `ollama-models/`, `tools/`, the ledger and the personal files).

## Release 2 (`0.3.0`): web UI and Windows installer

### U1. `bytx serve` skeleton
- [ ] Done · **Needs:** T10 · **PRD:** R2.1

**Files:** `pyproject.toml` (add `starlette`, `jinja2`, `uvicorn`, `python-multipart`), new `src/bytx/web/__init__.py`, `src/bytx/web/templates/base.html`, `src/bytx/web/static/htmx.min.js`, `src/bytx/cli.py` (`serve`), new `tests/test_web.py`.

**Do:**
- `create_app(cfg, worker) -> Starlette`, holding:
  - `HOST = "127.0.0.1"` as the only bind address;
  - `TrustedHostMiddleware(allowed_hosts=["127.0.0.1", "localhost"])`;
  - a check that rejects any POST whose `Origin` isn't the app's own;
  - `/healthz`, returning `{"app": "bytx"}`;
  - an Ollama-reachable check with a short timeout, cached for a few seconds.
- Vendor htmx 2.x with its license header. No CDN.
- `base.html` has the nav and a font stack that renders Bangla on stock Windows: `"Nirmala UI", "Noto Sans Bengali", system-ui`.
- `bytx serve [--port 8765] [--no-browser]`. If the port already answers `/healthz` as bytx, open the browser and exit, so a second click on the shortcut doesn't fail. Otherwise start the worker (U3), open the browser and run uvicorn.

**Done when:** no code path binds to anything but localhost.

**Tests:**
- `uvicorn.run` (monkeypatched) gets `host="127.0.0.1"`.
- `Host: evil.com` gets 400.
- A POST with a foreign `Origin` gets 403.
- `/healthz` returns 200.

### U2. Bilingual UI
- [ ] Done · **Needs:** U1 · **PRD:** R2.3

**Files:** new `src/bytx/web/i18n.py`, templates, `tests/test_web.py`.

**Do:**
- `STRINGS = {"bn": {...}, "en": {...}}` and `t(lang, key)`.
- A `lang` cookie, set through `GET /lang/{code}`, which redirects back. Bangla is the default.
- `<html lang=...>` follows the UI language.
- Transcript text is never passed through `t()`.

**Done when:** no visible UI string exists outside the two tables.

**Tests:**
- Both tables have identical key sets.
- Every template, after stripping `{{ }}`, `{% %}`, `{# #}` and HTML tags, contains no letters.
- The `placeholder`, `title`, `aria-label`, `alt` and `value` attributes contain no letters outside `{{ }}`.

### U3. Background worker
- [ ] Done · **Needs:** U1 · **PRD:** R2.1

**Files:** new `src/bytx/web/worker.py`, `tests/test_web.py`.

**Do:**
- One daemon thread with its own `store.connect` and a `kick()` that sets an event.
- On each wake it runs `pipeline.run` and records `running`, `last_result`, `stopped`, and the message from a caught `ModelsMissing`.
- `serve` kicks it once at startup, so a restart resumes the queue.

**Done when:** closing the browser doesn't stop a running job, and restarting `serve` resumes the queue.

**Tests:** with `pipeline.run` monkeypatched:
- one kick gives one run;
- kicks during a run give exactly one follow-up run;
- two runs never overlap;
- a `ModelsMissing` is recorded and doesn't kill the thread.

### U4. Progress reporting
- [ ] Done · **Needs:** U3 · **PRD:** R2.2 (Queue), and the user story about long CPU runs

**Files:** `src/bytx/pipeline.py` (`run`, `transcribe_one`), `src/bytx/transcribe.py` (`Transcriber.transcribe`), `src/bytx/web/worker.py`, tests.

**Do:**
- Add `on_progress: Callable[[str, str, float], None] | None = None` to `pipeline.run` and pass it through `transcribe_one` to `Transcriber.transcribe`.
- After each clip, `transcribe` reports seconds of audio done divided by the total.
- Each stage reports when it starts a video.
- The worker keeps the latest `(video_id, stage, fraction)`.

**Tests:** a fake transcriber with three clips reports increasing fractions that end at 1.0. Existing callers that pass nothing behave the same.

### U5. Add page
- [ ] Done · **Needs:** U2, U3 · **PRD:** R2.2 (1)

**Do:**
- A textarea for one or more URLs and a radio group `bn`/`en`/`auto` with nothing preselected.
- The submit button stays disabled until a language is chosen (a small inline `onchange`).
- The server validates the language again, with `langs.normalize` or `auto`.
- Call `pipeline.add_targets`, then `worker.kick()`, then redirect to Queue.

**Tests:**
- No language posted gets 400 and queues nothing.
- `bn` queues with `language="bn"`.
- `auto` queues with `language=None`.
- The page has no radio pre-checked.

### U6. Queue page
- [ ] Done · **Needs:** U4 · **PRD:** R2.2 (2)

**Do:**
- An htmx partial polled every 2 s (`hx-trigger="every 2s"`) shows each non-finished video with its status, the current video's stage and progress, `stopped`, and the models-missing message.
- A "retry failed" POST calls `pipeline.retry_failed`, then `worker.kick()`.

**Tests:** the partial renders each status and the progress. Retry moves failed videos back to their earliest unfinished stage.

### U7. Library page
- [ ] Done · **Needs:** U2 · **PRD:** R2.2 (3)

**Do:**
- The video list from `store.list_videos`.
- Keyword search through `store.search_fts`, with a `ValueError` shown as a message.
- Semantic search through `rag.semantic_search`, rendered only when Ollama is reachable.
- Hits link to the Video page at that timestamp.

**Tests:** keyword search renders hits. With Ollama absent there is no semantic control, and a direct request for semantic search gets 404.

### U8. Video page
- [ ] Done · **Needs:** U2, T4 · **PRD:** R2.2 (4)

**Do:**
- The transcript, with each timestamp linking to `platforms.timestamp_url(id, start_s)` (`target=_blank`, `rel=noopener`). The transcript block carries `lang=<video language>`.
- Quality flags from `quality.assess`.
- txt, srt and vtt downloads through `export.render` with a `Content-Disposition` header.
- Remove: a GET shows the `remove_video(apply=False)` preview, and a confirm POST calls `apply=True` and redirects to Library.

**Tests:** timestamp links are correct for YouTube and Facebook. Downloads have the right content. Remove only happens on the POST. An unknown id gets 404.

### U9. Ask page
- [ ] Done · **Needs:** U2 · **PRD:** R2.2 (5)

**Do:** a question form that calls `rag.ask`, with citations linking to the source at their timestamp. The nav link and the route only exist when Ollama is reachable; otherwise the route returns 404.

**Done when:** every page works with Ollama absent: Ask and semantic search are hidden.

**Tests:** with the Ollama check returning False, every page renders 200 and no page contains the Ask link. With a mocked client, an answer renders with its citation links.

### U10. `installer/install.ps1`
- [ ] Done · **Needs:** U1–U9 · **PRD:** R2.4

**Do:** a PowerShell script with `$ErrorActionPreference = 'Stop'` and `Start-Transcript` to `%TEMP%\bytx-install.log`:
1. If uv is missing, run the official installer (`irm https://astral.sh/uv/install.ps1 | iex`), then call `uv` by its full path (`$env:USERPROFILE\.local\bin\uv.exe`).
2. If `nvidia-smi -L` succeeds, the spec is `bytx[gpu]`; otherwise `bytx`. Run `uv tool install --upgrade --python 3.12 <spec>`. A re-run upgrades in place and never touches `~/bytx`.
3. With `-WithOllama`, if Ollama is not installed, download the official `OllamaSetup.exe` and run it silently. Check its silent flags first.
4. Refresh `$env:Path` from the user and machine registry values, then run `bytx setup`.
5. Create Desktop and Start-menu shortcuts (WScript.Shell) that run `<uv tool dir --bin>\bytx.exe serve`.

On error, print a Bangla and English message with the log path, then wait for a key press.

**Done when:** running it by hand on a clean VM reaches the Add page.

### U11. `installer/setup.iss` and the CI build
- [ ] Done · **Needs:** U10 · **PRD:** R2.4

**Do:**
- An Inno Setup script with `PrivilegesRequired=lowest` that ships `install.ps1`.
- A `[Tasks]` checkbox, "Also enable Q&A and summaries (about 5 GB download)", in Bangla and English, mapped to `-WithOllama`.
- `[Run]` runs `powershell -NoProfile -ExecutionPolicy Bypass -File "{tmp}\install.ps1"` and waits.
- Add a CI job on `windows-latest` for `v*` tags: install Inno Setup if the runner lacks it, `iscc installer/setup.iss`, and attach `setup.exe` to the GitHub Release (`permissions: contents: write`). The installer is unsigned.

**Done when:** a tag produces a release with `setup.exe` attached.

### U12. `docs/quickstart.bn.md`
- [ ] Done · **Needs:** U11 · **PRD:** R2.5

**Do:** in Bangla, cover downloading `setup.exe`, getting past SmartScreen (*More info → Run anyway*, with a screenshot the **maintainer** takes on the VM), the Q&A checkbox and its 5 GB cost, and the first transcript. Link it from the README and the release notes.

### U13. Release `0.3.0` (Maintainer + engineering)
- [ ] Done · **Needs:** U12 · **PRD:** R2.4, R2 P1

**Do:**
- On a fresh Windows 10 or 11 VM with no Python and no GPU, double-clicking `setup.exe` must end at the Bangla Add page with nothing typed.
- Re-run `setup.exe` and confirm it upgrades and keeps `~/bytx`.
- Run the tester round with 2–3 non-technical Bangladeshi users and fix what blocks them.
- Bump to `0.3.0` and tag `v0.3.0`.

---

## Order

```
S0 ──────────────────────────────┐
T1a → T1b → T2a → T2b → T2c      │
          ├→ T3a → T3b ──────────┤
          └→ T4                  │
T5a → T5b ─────────────→ T5c ←───┘ → T5d
T5a → T6
(all R1 code) → T7 → T8 → T9 → T10 → T11
T10 → U1 → U2 ─┬→ U5, U7, U8, U9
         └→ U3 → U4 → U6
U5–U9 → U10 → U11 → U12 → U13
```

Within release 1, follow the PRD's order when nothing else decides: data folder and defaults, then Ollama optional, then `remove`, then dependencies, then docs.

## Coverage: PRD requirement → tasks

| PRD | Tasks |
|---|---|
| R1.1 Data folder | T1b, T6 (smoke) |
| R1.2 Packaged defaults + overrides | T1a |
| R1.3 Pre-converted model; setup downloads only | T2a, T2b, T2c, T8 |
| R1.4 Ollama optional | T3a, T3b |
| R1.5 Dependencies and platforms | S0, T5a, T5b, T5c, T5d, T6 (macOS smoke) |
| R1.6 `bytx remove` | T4 |
| R1.7 CI and release | T6, T10 |
| R1.8 Docs and repo cleanup | T7 |
| R1 P1 | T9 |
| §7 Maintainer setup | T11 |
| R2.1 `bytx serve` | U1, U3 |
| R2.2 Pages | U4–U9 |
| R2.3 Bilingual UI | U2 |
| R2.4 Windows installer | U10, U11, U13 |
| R2.5 Bangla quick-start | U12 |
| R2 P1 Tester round | U13 |

## Deliberately left out

These are outside the PRD. Add them only if the maintainer asks.

- Removing the in-code `.get(key, default)` fallbacks in `config.py`. `defaults.toml` makes them redundant, but they're harmless.
- An uninstaller that removes the uv tool and shortcuts.
- Per-video retry. The Queue page retries all failed videos, which reuses `pipeline.retry_failed`.
- Pausing Ask while the GPU is transcribing.
- Renaming `yt-transcribed-files/`. It now holds Facebook transcripts too.
