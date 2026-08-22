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

### BUG-002 — ODE REST silently ignores unknown query parameters, returning unfiltered results

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** data-ingest
- **Symptom:** Querying LROC NAC at the Chandrayaan-3 site with `minincidenceangle=30&maxincidenceangle=60` and with `minincidenceangle=70&maxincidenceangle=89` both returned `Count = 390` — identical. So did `iangle1`/`iangle2`. 390 turned out to be the *unfiltered* count for the bounding box.
- **Root cause:** The parameter names were wrong, but ODE does not reject unrecognized query parameters — it drops them and answers the remaining query. A wrong filter name is indistinguishable from no filter. The real names are **`mininangle`** and **`maxinangle`** (likewise `minemangle`/`maxemangle`, `minphangle`/`maxphangle`), documented only in the ODE REST V2.1.6 User's Manual PDF, not on the landing page.
- **Fix:** Use `mininangle`/`maxinangle`. Manual cached at `data/interim/ode_manual.txt` for the full parameter list. Verified working: same bbox returns 0 / 30 / 333 for incidence 30-50 / 50-70 / 70-90, against 390 unfiltered.
- **Check:** `python scripts/survey_coverage.py --self-check` asserts that an incidence-filtered count is strictly less than the unfiltered count for a known bbox. Fails if a parameter name silently stops biting.
- **Why this one mattered:** it would not have crashed. It would have quietly built the entire Stage-B training set from images at whatever illumination happened to be there, while the logs claimed a controlled illumination split — and the Stage-B ablation, our central scientific claim, would have been measuring nothing. **Rule going forward: never trust a filter that was not observed to change the result count.**

### BUG-001 — Non-ASCII characters print as `?` in Windows console output

- **Date:** 2026-08-22
- **Status:** FIXED
- **Area:** env
- **Symptom:** `python scripts/check_env.py` printed `GDAL 3.10.3 ? 204 drivers`. The em-dash in the f-string came out as a replacement character.
- **Root cause:** Windows console defaults to the legacy `cp1252` code page, not UTF-8. Python encodes stdout with that codec, and any character outside cp1252 is mangled. Not a rasterio or GDAL problem at all — the driver probe itself was correct.
- **Fix:** Replaced the em-dash with an ASCII hyphen in `scripts/check_env.py:69`. Keeping console output ASCII-only is cheaper than forcing a UTF-8 code page on every machine that runs this.
- **Check:** `python scripts/check_env.py` output contains no `?` replacement characters.
- **Note:** This will recur with real force during data ingest. **PDS4 labels are UTF-8 XML** and may carry non-ASCII in target names, mission descriptions, and units. Read and write those files with an explicit `encoding="utf-8"` — never rely on the platform default — or the same mangling silently corrupts metadata rather than just cosmetics.
