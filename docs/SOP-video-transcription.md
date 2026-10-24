# SOP: Transcribing videos with bytx

Applies to anyone (human or AI agent) who is asked to "transcribe the new video(s) in `urls.txt`" or
similar. Follow it in order. Do not skip step 2 or step 5.

`bytx` here is the installed command (`uv tool install bytx`, or `uv tool install --editable ".[gpu]"`
from a clone). Your library lives in the data folder: `$BYTX_HOME` if set, else `~/bytx`. `bytx doctor`
prints where it is. Paths below are relative to it.

## Why this exists

On 2026-09-27 an English video was queued with a bare `bytx add urls.txt --run`. The config defaults
to `asr.default_language = "bn"`, so the video was recorded as Bangla and decoded with the Bangla ASR
model, which wrote English speech out as Bangla-script gibberish. Nothing flagged it: the pipeline
reported `failed=0`, the output was 154 words/min, and `bytx quality` showed no flags. The error was
only caught by the user reading the output. Two rules follow from that:

1. **The language of each video is decided by a person or agent before queueing, never by the default.**
2. **"The pipeline finished" is not "the transcript is right".** Verify the output (step 5) and say
   what you checked.

## Steps

### 1. Preflight

```powershell
bytx status          # anything stuck in queued/failed from an earlier run?
```

Run `bytx doctor` if this is a fresh machine or the last run had GPU/Ollama errors.
If videos are already `queued`, decide whether they belong to this job before running: `bytx run`
processes the **whole** queue, not just the URLs you just added.

### 2. Decide the language of every URL (mandatory)

For each URL in `urls.txt`:

| Question | Action |
|---|---|
| Already transcribed? (`TRANSCRIBED_LEDGER.md`, `bytx list`) | Skip it; `bytx add` will refuse it anyway. To redo, see "Redoing a video". |
| Language obvious? Title, channel and description are English → `en`; Bangla script or a Bangla-speaking creator talking Bangla → `bn`. | Use that. |
| Not obvious (mixed title, Banglish, unfamiliar channel) | Use `--lang auto`, or ask the user. Do not guess. |

- Bangla speech with English words mixed in is still `bn`. Do **not** use Whisper language ID or
  YouTube's language metadata to decide: both mislabel that case (see CLAUDE.md).
- Fetch the title cheaply if you need it:
  `uvx --from "yt-dlp[default,deno]" yt-dlp --skip-download --print "%(channel)s | %(title)s" <url>`
- Never fall back to `asr.default_language`. If you cannot decide and cannot ask, use `--lang auto`.
- If `urls.txt` mixes languages, split it into one temporary file per language (a scratch folder, not
  the data folder or the repo), or use `--lang auto` for the whole file.

State your decision in your reply before running, e.g. "J-5L2k9LHAc: English (title/channel) → `--lang en`".

### 3. Queue with an explicit `--lang`

```powershell
bytx add urls.txt --lang en          # or bn / auto. Always pass --lang.
```

Check the output line `Queued N new video(s).` matches the number of *new* URLs you expected. Fewer
means some were already in the DB or ledger; find out which, do not assume.

### 4. Run

```powershell
bytx run
```

(`bytx add --run` does add + run in one go; fine once step 2 is done.) Long videos take a while, so run it
in the background and wait for completion. Ctrl+C is safe: `bytx run` resumes. If it ends with "Ollama
(optional) isn't running", the transcripts are saved but not indexed or summarized; start Ollama and
run `bytx run` again before step 5.

### 5. Verify (do not skip)

For **each** new video id (`bytx list` shows them):

1. **Status and language.** `bytx list --status summarized`: the row is present and the `Lang`
   column is what you decided in step 2 (for `--lang auto`, note which language it picked).
2. **Script matches language.** `bytx show <id>` and read the first ~10 lines and a few from the
   middle:
   - `en` → English words in Latin script, real sentences. Bangla characters or gibberish = wrong.
   - `bn` → Bangla script. Mostly Latin text for a Bangla video = wrong (possible `auto` mis-detect).
3. **Quality flags.** `bytx quality <id>`. Note this does **not** detect a wrong-language transcript
   (that is why step 5.2 exists). It catches gaps, low-confidence clips and abnormal words/min.
4. **Summary exists and is in sensible language.** `data/videos/<id>/summary.md`.
5. **Artifacts.** A new folder under `yt-transcribed-files/<date>/`, a new row in `yt-transcribed-files/INDEX.md`,
   a new entry in `TRANSCRIBED_LEDGER.md`.

If any check fails, go to "Redoing a video". Do not report success.

### 6. Report

Tell the user, per video: id, title, language used and why, the folder path, and *what you checked* in
step 5 (e.g. "first 10 lines are coherent English; no quality flags"). Include any failures or flags
verbatim. `failed=0` alone is not a report.

### 7. Files

Transcripts, `INDEX.md` and `TRANSCRIBED_LEDGER.md` live in the data folder, outside the code repo, so
there is nothing to commit. Do not touch `urls.txt` (it is the user's input file).

## Redoing a video

Use when the language was wrong or the transcript is unusable. `bytx add` refuses anything in the DB or
ledger, so all of it has to be cleared first. `bytx remove` does that and rebuilds `INDEX.md`:

```powershell
bytx remove <video_id>          # dry run: shows what would be deleted
bytx remove <video_id> --yes    # delete DB row, data/videos/<id>, transcript copy, ledger entry
bytx add <url> --lang <bn|en> --run
```

This deletes data, so confirm with the user first unless they already asked for the redo. Then repeat
step 5 on the new output.

## Failure quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Bangla-script gibberish for English speech | Video queued as `bn` | Redo with `--lang en` |
| English text for Bangla speech | Queued as `en`, or `auto` mis-detected | Redo with `--lang bn` |
| `Queued 0 new video(s)` | Already in DB/ledger | `bytx list`; redo only if intended |
| `bytx run` says "Run `bytx setup`" | ASR or Ollama models not downloaded | `bytx setup`, then `bytx run` |
| Status `failed` | Download/ASR/Ollama error | `bytx status` for the message, fix cause, `bytx retry` then `bytx run` |
| `bytx run` did more than you queued | Older videos were still `queued` | Check `bytx status` in step 1 next time |
| GPU / cuDNN / Ollama errors | Environment | `bytx doctor` |
| Facebook URL fails | No subtitles/playlists on FB; cookies may be needed | See README, `[fetch]` in `config.toml` |

## Rules for AI agents specifically

- Never run `bytx add` or `bytx add --run` without `--lang`.
- Never treat exit code 0 or `failed=0` as verification.
- Ask before deleting anything (redo), unless the user already asked for it.
- If you notice a mistake after the fact, say so plainly and propose the fix; do not quietly patch it.
