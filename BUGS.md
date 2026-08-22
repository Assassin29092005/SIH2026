# BUGS.md

Running bug log for the SIH26166 lunar image correspondence project.

## Rule

**Every bug found gets an entry here, including how it was resolved.**

Applies to all of them — crashes, silent wrong output, bad data assumptions, environment and dependency breakage, metric bugs. A bug that is fixed must have a filled-in **Fix**. A bug still open says `OPEN` in the status and gets updated when it is fixed.

Newest entries at the top of the log.

## Entry format

```markdown
### BUG-NNN — <one-line symptom>

- **Date:** YYYY-MM-DD
- **Status:** OPEN | FIXED
- **Area:** data-ingest | projection | stage-a-localize | stage-b-photometric | stage-c-matching | stage-d-fit | eval | env
- **Symptom:** what was observed, including the exact error line if there was one
- **Root cause:** what was actually wrong, not what it looked like
- **Fix:** what changed, and where (`file.py:line`)
- **Check:** the runnable thing that now fails if this regresses
```

Rules for writing entries:

- **Symptom and root cause are separate fields and usually differ.** If they read the same, the root cause has not been found yet.
- Quote the shortest decisive line of an error, not the whole traceback.
- A fix without a **Check** is incomplete for anything non-trivial. The check is the smallest thing that fails if the logic breaks again.
- Record bugs caused by wrong assumptions about the data (wrong units, wrong projection, flipped axis, mislabelled illumination angle) with the same weight as code bugs. On this project those are the expensive ones.

---

## Log

_No entries yet. First bug goes here._
