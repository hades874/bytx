# PRD: bytx on other people's computers

| | |
|---|---|
| Status | Approved 2026-10-22 (grilling session, Q1–Q27) |
| Owner | Hasan Farabi (maintainer) |
| Releases | **R1 = `0.2.0`** (installable package, technical users) · **R2 = `0.3.0`** (local web UI + Windows installer, non-technical users) |
| Audience | The maintainer, contributors, and coding agents working in this repo |

## 1. Problem statement

bytx works today, but only for one person: someone on Windows with an NVIDIA GPU who clones the repo and runs it from inside the repo folder. That person also has to install uv, ffmpeg, Deno and Ollama by hand, and torch for a one-time model conversion. A `pip`/`uv tool` install fails right away with "Config file not found", because `config.py` finds `config.toml` by walking up from the current directory to `pyproject.toml`. The people who most need Bangla transcription (students, researchers and creators in Bangladesh, often on laptops without NVIDIA GPUs and on slow or metered connections) can't use it at all. The maintainer's own library (45 videos) also lives inside the code repo, so the maintainer never exercises the path users would take.

## 2. Goals

1. **Installable anywhere:** on a fresh Windows, Linux or macOS machine, `uv tool install bytx` → `bytx setup` → `bytx add <url> --lang bn --run` produces a transcript with no clone, no torch and no manual ffmpeg or Deno install.
2. **Right-sized download:** users without an NVIDIA GPU never download the about 1.3 GB of CUDA libraries, and users without Ollama never download about 5 GB of LLM models, yet transcription and keyword search fully work for them.
3. **No terminal for non-technical users (R2):** a Bangladeshi Windows user double-clicks `setup.exe` and ends up at a Bangla-language browser page where they paste a link and get a transcript.
4. **Zero marginal cost per user:** all compute runs on the user's machine. Hosting is PyPI, Hugging Face and GitHub Releases, all free.
5. **The maintainer dogfoods it:** the maintainer's daily `bytx` is an install like any user's (editable), reading a library outside the repo, and the repo contains only code and docs.

## 3. Non-goals

| Non-goal | Why |
|---|---|
| Hosted web service | YouTube blocks yt-dlp from cloud IPs, GPU cost grows per user, and downloading on strangers' behalf carries legal exposure. |
| User accounts, multi-user, LAN access | `bytx serve` has no login, so it binds to `localhost` only. |
| Settings page in the UI | `config.toml` overrides are enough, and a settings page doubles the surface to maintain. |
| Auto-updater, telemetry | Upgrades are `uv tool upgrade bytx` or re-running `setup.exe`. No data leaves the machine. |
| Code signing (for now) | Costs money every year. An unsigned `setup.exe` plus SmartScreen instructions in Bangla is good enough to start. |
| A smaller, faster Bangla model for CPU (for now) | Accuracy matters more than speed for transcripts. Revisit only if users report CPU speed stops them (see P2). |
| Auto-detecting language with Whisper LID or YouTube metadata | This is an existing rule: both mislabel Bangla mixed with English. The language is always an explicit choice. |
| Hindi or other languages | Out of scope for this effort. |

## 4. Personas

- **P-Tech, technical user (R1):** comfortable with a terminal and a README, on Windows, Linux or macOS, with or without an NVIDIA GPU.
- **P-BD, non-technical Bangladeshi user (R2, the primary audience):** a Windows laptop, often without an NVIDIA GPU, a slow or metered connection, prefers a Bangla UI, won't open PowerShell.
- **P-Max, the maintainer:** Windows + RTX 4070, an existing 45-video library, a Claude-driven transcription SOP, keeps tools and caches on `D:\`.

## 5. User stories

**P-Tech**
- As a technical user, I want to install bytx with one `uv tool install` command and run it from any folder, so that I don't have to clone the repo or remember where it lives.
- As a technical user without an NVIDIA GPU, I want a CPU-only install, so that I don't download 1.3 GB of CUDA libraries I can't use.
- As a technical user with an NVIDIA GPU, I want `bytx doctor` to tell me when I'm missing the GPU extra, so that I don't unknowingly transcribe at CPU speed.
- As a technical user without Ollama, I want transcription and keyword search to work and the LLM features to explain what's missing, so that I'm not left with videos marked `failed`.
- As a technical user who queued a video with the wrong language, I want `bytx remove <id>`, so that I can re-add it correctly without hand-editing the DB, folders and ledger.
- As a technical user, I want my settings file to hold only the settings I changed, so that upgrades bring new defaults.

**P-BD**
- As a non-technical user, I want to download and double-click one installer, so that I never type commands.
- As a non-technical user, I want the app in Bangla, so that I understand every button and message.
- As a non-technical user, I want to paste a link and choose its language, so that the transcript comes out in the right script.
- As a non-technical user, I want to watch progress and retry failures, so that I know whether a long CPU run is still working.
- As a non-technical user, I want to click a line in a transcript and jump to that moment in the video, so that I can check what was said.
- As a non-technical user on a metered connection, I want Q&A and summaries to be an optional extra, so that the base install stays smaller.

**P-Max**
- As the maintainer, I want my library outside the repo, so that the public repo holds only code and I use bytx the same way users do.
- As the maintainer, I want my daily `bytx` to run my working copy, so that I test changes on my real library immediately.
- As the maintainer, I want tagged releases to publish themselves after tests pass on Windows and Linux, so that a broken build never reaches PyPI.

## 6. Requirements

### Release 1 (`0.2.0`): installable package, P0

**R1.1 Data folder (`~/bytx`, override `BYTX_HOME`).**
Holds the user's `config.toml` (overrides only), `library.db` with `videos/`, the readable transcripts folder, `TRANSCRIBED_LEDGER.md` and `glossary.txt`. Models are not stored here. Resolution order: `--config` → `BYTX_HOME` → `~/bytx`. Delete the walk-up `_find_project_root` (`src/bytx/config.py:335`). `ledger.ledger_path` (`src/bytx/ledger.py:25`) follows the new root.
- [ ] `bytx status` works from any directory after a `uv tool install`.
- [ ] On a machine with no `~/bytx`, the first command creates it. No error, no prompt.
- [ ] Setting `BYTX_HOME` moves everything. `--config` still wins.
- [ ] Tests cover all three resolution paths.

**R1.2 Packaged defaults plus user overrides.**
The current `config.toml` ships inside the package as the defaults. The user's file is deep-merged over it (tables merge, scalars replace).
- [ ] A missing or empty user `config.toml` runs on the defaults.
- [ ] A user file containing only `[asr]\ndevice = "cpu"` changes only that setting.
- [ ] A default changed in a new release reaches users who didn't override it.

**R1.3 Pre-converted Bangla model on Hugging Face; `setup` downloads only.**
Publish `bytx/tugstugi-medium-bn-ct2`: a float16 CTranslate2 conversion of `bengaliAI/tugstugi_bengaliai-asr_whisper-medium`, Apache-2.0, with a model card crediting bengaliAI. The default config points `asr.models.tugstugi-medium-bn` at it with `already_ct2 = true`. `bytx setup` downloads every configured ASR model, plus the Ollama models when Ollama is reachable. It no longer needs `scripts/convert_model.py` (which today breaks installed wheels, `src/bytx/cli.py:167`). Reuse the `already_ct2` download branch of `scripts/convert_model.py:convert_or_download`, moved into the package. The `convert` extra and the script stay as maintainer tools.
- [ ] A fresh install without the `convert` extra completes `bytx setup`.
- [ ] `bytx run` before `bytx setup` stops with "run `bytx setup`", and every video stays `queued` (none marked `failed`).

**R1.4 Ollama is optional.**
`pipeline.run` (`src/bytx/pipeline.py:303-320`) checks Ollama once before the index stage. If it's unreachable, the run stops after transcription with a one-line hint. `doctor` reports Ollama as optional (a warning, not a failure). `setup` skips the Ollama pulls when Ollama is absent. `ask`, `search --semantic` and `summarize` print an "install Ollama to enable this" message.
- [ ] With Ollama stopped, `bytx add <url> --lang bn --run` leaves the video `transcribed`, `failed=0`, and `show`, `export`, keyword `search` and `quality` work.
- [ ] Starting Ollama and running `bytx run` again moves the video through `indexed` and `summarized`.

**R1.5 Dependencies and platforms.**
- `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` move to a `gpu` extra, limited to Windows and Linux. Plain `bytx` is CPU-only.
- `ensure_cuda_dll_paths` (`src/bytx/cuda_dll.py:41`) must tolerate the missing `nvidia` namespace. `find_spec("nvidia.cublas")` raises `ModuleNotFoundError` when the parent package is absent, which is exactly the CPU-only Windows install.
- `yt-dlp[default,deno]` replaces the manual Deno install.
- ffmpeg is no longer a user requirement. The pipeline never calls it (`src/bytx/fetch.py:216-226` downloads raw `bestaudio` and PyAV decodes it). `doctor` stops checking for it, and `bench` documents that it needs ffmpeg.
- On a non-`[gpu]` install, `doctor` runs `nvidia-smi` and, if it finds a GPU, suggests `uv tool install "bytx[gpu]"`.
- When ASR runs on CPU, `WhisperModel` gets `cpu_threads` = physical cores (`src/bytx/transcribe.py:253`; the default today is 4).
- [ ] A plain install on Windows downloads no `nvidia-*` wheel and transcribes on CPU without crashing.
- [ ] `bytx[gpu]` on the RTX 4070 runs `cuda/float16` (`doctor` shows no fallback note).
- [ ] The install succeeds on macOS (CPU).
- [ ] One real YouTube and one real Facebook download plus transcription succeed with ffmpeg off PATH and no system Deno. **This check gates dropping ffmpeg.**
- [ ] CPU real-time factor on a Ryzen 5 5600 is re-measured with `bytx bench` against the 2.25 baseline.

**R1.6 `bytx remove <id>`.**
Port `scripts/redo_video.py` into the CLI with the same behaviour. A dry run by default; `--yes` deletes the DB row (segments, chunks and summary cascade), `videos/<id>/`, its readable-transcripts copy (then rebuilds `INDEX.md`), and its ledger entry. Delete the script.
- [ ] The dry run lists exactly what would be removed and changes nothing.
- [ ] After `--yes`, `bytx add` accepts the same URL again.
- [ ] An unknown id exits non-zero with a clear message.

**R1.7 CI and release.**
GitHub Actions run `pytest` and `ruff check` on `windows-latest` and `ubuntu-latest` for every push and PR. A `v*` tag builds and publishes to PyPI through trusted publishing (no stored token).
- [ ] Tagging `v0.2.0` produces a PyPI release that `uv tool install bytx` installs.
- [ ] A failing test blocks publishing.

**R1.8 Docs and repo cleanup.**
- README: install (`uv tool install bytx` / `"bytx[gpu]"`), the data folder and `BYTX_HOME`, Ollama as optional, CPU speed expectations (about 2.25× the audio's length on a 6-core desktop CPU, slower on laptops), macOS CPU-only, a Linux GPU `LD_LIBRARY_PATH` note marked community-tested, the glossary format, and a responsible-use note (respect copyright and platform terms).
- `CLAUDE.md` and `docs/SOP-video-transcription.md` are rewritten for the installed `bytx`: no `env.ps1`, no `uv run`, and `bytx remove` replaces the redo script.
- `env.ps1`, `urls.txt` and `glossary.txt` (personal) leave the repo.
- [ ] A README-only walkthrough on a fresh machine reaches a transcript.

### Release 1, P1

- `bytx setup` prints download sizes before starting.
- A Bangla README quick-start for P-Tech users who prefer Bangla.

### Release 2 (`0.3.0`): local web UI and Windows installer, P0

**R2.1 `bytx serve`.**
Starlette with Jinja2 templates and htmx. htmx is vendored into the package so the UI works offline. It binds to `127.0.0.1` only, opens the browser, and runs `pipeline.run` in one background worker thread (one job at a time, one GPU). It reuses `store`, `pipeline`, `rag` and `export` as they are.
- [ ] No code path binds to anything other than localhost.
- [ ] Closing the browser doesn't stop a running job. Restarting `serve` resumes the queue (the pipeline is already resumable).

**R2.2 Pages (exactly these).**
1. **Add:** paste one or more URLs and choose `bn`, `en` or `auto`. No language is pre-selected, and submit stays disabled until one is chosen.
2. **Queue:** live status per video, with retry for failed videos.
3. **Library:** the video list, keyword search, and semantic search only when Ollama is present.
4. **Video:** the transcript with each timestamp linking to that moment in the source, the quality flags, txt/srt/vtt download, and remove (with confirmation).
5. **Ask:** shown only when Ollama is reachable. Answers cite timestamps that link to the source.
- [ ] Each page works with Ollama absent (Ask is hidden, semantic search is hidden).

**R2.3 Bilingual UI.**
Every visible string lives in one table per language (`bn`, `en`). Bangla is the default, and the switch is remembered in a cookie. Transcript text is never translated or modified.
- [ ] No hard-coded UI string outside the tables (a test enforces it).

**R2.4 Windows installer.**
`install.ps1` (in the repo, served from GitHub raw):
1. installs uv with its official installer;
2. runs `uv tool install bytx`, or `"bytx[gpu]"` if `nvidia-smi` finds a GPU;
3. installs Ollama with the official installer if the user chose it;
4. runs `bytx setup`;
5. creates Desktop and Start-menu shortcuts that run `bytx serve`.

`setup.exe` is Inno Setup wrapping `install.ps1`, with a checkbox: "Also enable Q&A and summaries (about 5 GB download)". CI builds it on each tag and attaches it to the GitHub Release. It's unsigned.
- [ ] On a fresh Windows 10/11 VM with no Python and no GPU, double-clicking `setup.exe` ends at the Bangla Add page without typing anything.
- [ ] Re-running `setup.exe` upgrades in place and keeps `~/bytx`.

**R2.5 Bangla quick-start.**
`docs/quickstart.bn.md`: download, getting past SmartScreen (*More info → Run anyway*, with a screenshot), first transcript.

### Release 2, P1

- A tester round with 2–3 non-technical Bangladeshi users before announcing `0.3.0`.

### P2: future considerations (design so these stay possible)

- A smaller Bangla model chosen automatically on CPU, if users report speed problems. Keep model choice per language in config, as today.
- Code signing for `setup.exe`.
- Access from other computers on the LAN, behind a login. Keep the bind address in one place.
- An update notice in the UI.

## 7. The maintainer's own setup (part of R1)

1. Run `uv tool install --editable ".[gpu]"` from the clone, so `bytx` is on PATH everywhere and runs the working copy.
2. Set permanent user environment variables to replace `env.ps1`: `BYTX_HOME` (a folder on `D:\`), `HF_HOME`, `OLLAMA_MODELS` (today `ollama-models/` in the repo), and `UV_CACHE_DIR` / `UV_PYTHON_INSTALL_DIR` if uv should keep its caches on `D:\`.
3. Look at the target first, then copy `data/` (15 MB), `yt-transcribed-files/` (4.6 MB), `TRANSCRIBED_LEDGER.md`, `glossary.txt` and `urls.txt` into `BYTX_HOME`.
4. Verify: `bytx status` shows 45 videos (39 bn, 6 en, all `summarized`), `bytx search` returns hits, and `bytx ask` answers with citations.
5. Only after verifying, and after asking, delete the old copies from the repo folder.

## 8. Success metrics

No telemetry, so only public counters and release checklists are used.

**Leading (checked every release)**
| Metric | Target |
|---|---|
| Fresh-VM install checklist (Windows, Ubuntu, macOS CPU) reaches a transcript using only README steps | Pass on all three before tagging |
| `setup.exe` fresh-VM check (R2.4) | Pass before tagging `0.3.0` |
| CI on Windows + Linux | Green on every tag |
| Install-failure GitHub issues in the first 30 days after a release | ≤ 3, each fixed in a patch release |

**Lagging (initial hypotheses, revisit 3 months after each release)**
| Metric | Source | Target |
|---|---|---|
| PyPI downloads | pypistats | 100/month by 3 months after `0.2.0` |
| `setup.exe` downloads | GitHub Release asset counts | 50 in the first month after `0.3.0` |
| Bangla model downloads | Hugging Face model page | Tracks PyPI + installer downloads (a gap means setup is failing) |
| Maintainer dogfooding | Repo state | No personal data in the repo, and every daily run goes through the installed `bytx` |

## 9. Open questions

| Question | Owner | Blocking? |
|---|---|---|
| Create a Hugging Face account and a `bytx` organisation (the name was free on 2026-10-22). | Maintainer | Blocks R1.3 |
| Claim `bytx` on PyPI and configure the trusted publisher for this repo (the name was free on 2026-10-22). | Maintainer | Blocks R1.7 |
| Do real YouTube and Facebook downloads work with ffmpeg off PATH, and does yt-dlp find the pip-installed Deno? **Partly answered 2026-10-22 (spike S0):** with neither ffmpeg nor Deno on PATH (the venv's Scripts folder off PATH, as in a `uv tool` install), yt-dlp reports `JS runtimes: deno-2.9.8`, so it finds the pip Deno by itself. A YouTube `bestaudio` download (webm/opus, 265 s) decoded with PyAV, no ffmpeg. Facebook is not checked yet: the library has no Facebook video. | Engineering | Facebook part only: checked in T5d |
| Do Linux GPU installs find the pip CUDA libraries without `LD_LIBRARY_PATH`? There's no Linux GPU machine to test on. | Community / engineering | No (documented as community-tested) |
| During the maintainer migration, reuse the existing local `models/` (3 GB) or download again from Hugging Face? | Engineering | No |
| Responsible-use wording in the README. | Maintainer | No |
| Measured first-time download total (plain, `[gpu]`, with Ollama) for the README. | Engineering | No |

## 10. Phasing and dependencies

No hard deadlines. Order:

1. **Code-only R1 work:** R1.1 data folder + R1.2 defaults → R1.4 Ollama optional → R1.6 `remove` → R1.5 dependencies (including the `cuda_dll` fix and the ffmpeg-off check) → R1.8 docs.
2. **Needs the maintainer:** Hugging Face account and organisation → model card and `hf upload` command prepared → the maintainer uploads with their own token → default config switched (R1.3).
3. **Release:** PyPI trusted publisher → R1.7 CI → fresh-VM checklist → tag `v0.2.0`.
4. **Maintainer migration** (section 7).
5. **R2:** R2.1 serve → R2.2 pages → R2.3 bilingual → R2.4 installer + CI build → R2.5 quick-start → tester round → tag `v0.3.0`.

## 11. Decision log and measured facts

| # | Decision |
|---|---|
| Q1 | Purpose: open source / give away (not a team tool, not revenue) |
| Q2 | First users: technical, then non-technical Windows users |
| Q3 | No servers; everything local |
| Q4 | Maintainer's library moves to a per-user data folder |
| Q5 | Distribute on PyPI (`uv tool install bytx`) |
| Q6 | Officially Windows + Linux (NVIDIA), best-effort CPU-only on any OS including macOS |
| Q7 | Ollama optional |
| Q8 | Publish a pre-converted Bangla model on Hugging Face |
| Q9 | Packaged defaults + user overrides |
| Q10 | Maintainer uses an editable install; `env.ps1` leaves the repo |
| Q11 | Web UI in release 2 |
| Q12 | CI + PyPI trusted publishing |
| Q13 | Explicit `bytx setup` (the primary users are Bangladeshi, so the Bangla model is always needed) |
| Q14/Q19 | Hugging Face isn't set up yet; use a `bytx` organisation |
| Q15 | Data folder `~/bytx` |
| Q16 | `bytx remove` command |
| Q17 | Deno via the yt-dlp extra, drop ffmpeg (after verification), personal files out, keep `CLAUDE.md`/SOP, Linux GPU documented only |
| Q18 | UI: Starlette/FastAPI + Jinja + htmx |
| Q20/Q25 | CUDA libraries are an opt-in `[gpu]` extra; the installer picks it automatically |
| Q21 | `setup` downloads every configured model |
| Q22 | Bilingual UI, Bangla default |
| Q23 | Pages: Add, Queue, Library, Video, Ask; localhost only |
| Q24 | `install.ps1` + unsigned Inno Setup `setup.exe` |
| Q26 | Ollama is an installer checkbox |
| Q27 | CPU users get the same Bangla model, documented speed, and the thread fix |

Measured 2026-10-22:
- **CPU speed:** `tugstugi-medium-bn` on CPU int8 (Ryzen 5 5600, default 4 threads) took 404.5 s for a 180 s clip, a real-time factor of 2.25. No clips hit the token cap.
- **CUDA wheels (Windows):** cublas 528 MB + cudnn 712 MB + nvrtc 73 MB.
- **Licenses:** the Bangla source model is Apache-2.0; `deepdml/faster-whisper-large-v3-turbo-ct2` is MIT.
- **Library:** 45 videos; `data/` 15 MB, `yt-transcribed-files/` 4.6 MB, `models/` 3.0 GB.
