# Log: distribution work

Append-only, newest at the bottom. What's open lives in the [Status](PLAN-distribution.md#status) table of the plan; this file records what happened: checks run by hand, measurements, maintainer actions and decisions, with the commit when there is one. Commit ids are the ones after the 2026-10-25 message rewrite.

## 2026-10-22

- PRD approved; plan written (`ff15ad2`).
- Built and tested T1a, T1b, T2a+T2b, T2c, T3a, T3b, T4, T5a, T5b and T5c, one commit each (`d976b49` … `26556c3`).
- S0 spike (`71d8ee7`): with neither ffmpeg nor Deno on PATH, yt-dlp found the pip Deno (`deno-2.9.8`) and a YouTube `bestaudio` download (webm/opus, 265 s) decoded with PyAV. Facebook not checked: the library has no Facebook video. Details in PRD §9.
- T6 workflow written and its YAML checked; not run, because the branch wasn't pushed (`4bef936`).
- T7 docs for the installed `bytx`; `env.ps1`, `urls.txt` and `glossary.txt` untracked (`5a12feb`).
- T8 prep: model card written, upload commands checked against `hf` 1.32 (`7663893`).
- T9: `setup` prints download sizes first, checked live (1.62 GB for the English model); Bangla quick-start written (`7d934c3`).
- T5d, partial (`96e08ec`): CPU re-measured on the Ryzen 5 5600, `UkLJpcuJOIA` 1:00–4:00: RTF 0.90 at 4 threads, 0.87 at 6. The 2.25 baseline was not reproduced. Numbers in PRD §11.

## 2026-10-25

- Review fixes (`2f9c0a0`): a relative model folder resolves against the data folder; a `source` that is neither a folder nor a repo id says to fix it; `setup` pulls with `OLLAMA_HOST` from the config and gives a hint when the `ollama` command is missing; `ledger.remove_entry` swaps in a written copy; `doctor` and `remove` escape paths; install commands pin `--python 3.12`. Suite: 547 passed, ruff clean.
- Decision (maintainer): no Claude attribution in commit messages. The 18 commits on `distribution` were rewritten to drop it; code unchanged. The old history is in the local branch `backup/distribution-before-msg-rewrite`.
- `distribution` pushed for the first time. CI run [40119918337](https://github.com/hades874/bytx/actions/runs/40119918337) green: `test` on Windows and Ubuntu, `smoke` on Windows, Ubuntu and macOS; `publish` skipped (not a tag). T6 ticked (`1680610`).
- Added the Status table to the plan and started this log. T10 now merges `distribution` into `master` before tagging.
- T5d, Facebook: with ffmpeg, Deno and the venv's Scripts folder off PATH, `bytx add --lang bn --run` downloaded and transcribed `fb_212219056950214` (BBC News বাংলা, Wi-Fi tips, 7:12; chosen by the maintainer). Transcript read: coherent Bangla on topic, minor mis-hearings, no quality flags. Along the way, `run` refused to start until the English model was found, and stopped cleanly when Ollama wasn't running (video stayed `transcribed`, `failed=0`). Another BBC video (`480974782542443`) failed in yt-dlp 2026.08.19's Facebook extractor ("Cannot parse data"): a yt-dlp issue, not bytx.
- T5d, CPU: measured three clips on the Ryzen 5 5600 (int8, 6 threads, 1:00–4:00): RTF 1.01, 1.72 and 7.91, then 7.38 on a re-run of the slow one. One benchmark download got a one-off YouTube 403; a retry worked.
- Diagnosis: the slow clip is faster-whisper's temperature retries (compression ratio > 2.4), which dense Bangla speech triggers without repeating: 9 of 21 clips retried through temperature 1.0 and still failed. Same 7× cost on the GPU (136 s against 19 s). A first guess, memory pressure, was ruled out by logging free RAM during the re-run. Numbers in PRD §11.
- Decision (maintainer): fix the retries first (new task T5e), then write the README speed figure. T5d ticked.
- Found, not scheduled: `fetch.download_audio` returns a path when the download failed; the CPU fallback warning says "requested cuda" when the config asked for `cpu`. The repo's `tools/hf-cache` has no files for the English model, so T11's `bytx setup` will download it (1.62 GB).

## 2026-11-03

- Maintainer machine, until T11: `env.ps1` doesn't set `BYTX_HOME`, so `bytx` used an empty `~/bytx` (created by the first command, left in place) and found no models. Worked around with `BYTX_HOME` set to the repo and a gitignored repo-root `config.toml` pointing both `source`s at `models/`. `doctor`: all checks pass.
- Transcribed `fb_1643578627429139` (Revolution Education and Consultancy, Canada provinces, 13:56) with `--lang auto`: detected `bn`, correct (Bangla grammar with heavy English code-switching; English stretches come out as Bangla-script transliteration). No quality flags, 212 words/min. The library now has 46 videos, so T11's check of 45 is out of date.
- Found, not scheduled: Facebook titles from yt-dlp start with "13K views · 131 reactions | " and end with the uploader's name after another " | ". The prefix ends up in the export folder name, `INDEX.md` and the summary heading.

## 2026-11-06

- T8: the maintainer created the Hugging Face organisation `bytx` and uploaded `models/tugstugi-medium-bn` and the model card to [`bytx/tugstugi-medium-bn-ct2`](https://huggingface.co/bytx/tugstugi-medium-bn-ct2) (Hub commits `56d54a4`, `e5fd069`), logged in through `hf auth login`'s browser flow. The six files on the Hub match the local ones byte for byte (sha256 for `model.bin`, git blob ids over raw bytes for the rest).
- `defaults.toml` now points the Bangla model at the new repo with `already_ct2 = true`; a test checks every shipped model is `already_ct2`. Suite: 548 passed, ruff clean.
- Fresh-cache check: with empty `HF_HOME` and `BYTX_HOME`, no token, `models.download` fetched the Bangla model (1,531,332,891 bytes, as `download_size` reported) and `models.missing` then listed only `large-v3-turbo`. The English model was not re-downloaded; T11's `bytx setup` covers both end to end.
- Found: `bytx` on PyPI was taken on 2026-11-03 by an unrelated project (BYTtrade's crypto-exchange SDK, 0.1.2–0.1.5). No pending publisher had been created, and one wouldn't have reserved the name.
- Decision (maintainer): publish as `bytx-transcribe`; the command, import package, data folder, Hugging Face organisation and repo stay `bytx`. `pyproject.toml` sets `module-name = "bytx"` for `uv_build`; install commands in both READMEs, the SOP, the plan, the PRD and the `doctor` GPU hint now say `bytx-transcribe`. The built wheel is `bytx_transcribe-0.1.0-py3-none-any.whl` with the `bytx/` package and the `bytx = bytx:main` entry point. Suite: 548 passed, ruff clean.
- CI run [42208202846](https://github.com/hades874/bytx/actions/runs/42208202846) green on the rename (`3415b74`): the `bytx_transcribe` wheel installs and runs `bytx` on Windows, Ubuntu and macOS.
- Maintainer: registered the PyPI pending trusted publisher for `bytx-transcribe` (GitHub `hades874/bytx`, workflow `ci.yml`, environment `pypi`). Decision: no pre-release to hold the name; the first upload is the `v0.2.0` tag in T10, accepting that the name is unreserved until then.
