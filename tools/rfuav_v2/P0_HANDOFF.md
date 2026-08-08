# P0 — Box verification handoff (data-leakage remediation)

**Status when this was written:** all Mac-side edits + unit tests are done and green
(356-test suite passes, 24 new tests included). Nothing has been run at scale — the
real data lives only on the box (`E:\Drone_dataset`). This document is the turnkey
protocol to *verify* the P0 remediation there. **Do not treat any leakage number as
real until this chain has run on the box.**

Plan of record: `~/.claude/plans/eventual-toasting-salamander.md`.
ERRATA to publish after the run: E-016 (81→27 sessions, `all_roots` grouped_cv
66.7 %→33.3 %).

---

## 0. Red lines (non-negotiable, re-stated so the box operator sees them here)

- **Originals are READ-ONLY.** Every script in this chain writes **only** under
  `E:\Drone_dataset\RFUAV-v2\`. Rollback of the whole run = delete that directory.
  `quarantine-never-delete` for anything moved out of the canonical tree.
- **Never kill python by pattern on the shared box.** `PID 33888` is another user's
  `train_open_convlstm.py`. Check `CommandLine` before touching any PID.
- `unrar l` / `zipinfo -1` = **LIST ONLY, NEVER EXTRACT**.
- **complex64 invariant** is untouched this batch — do not let anything upcast
  `channels`/`normalize_power` to complex128.
- ScienceDB credentials are for ScienceDB login only; **never** write them into any
  file, log, commit, or saved command output.
- Never hide destructive stderr with `2>/dev/null`.

---

## 1. Precondition check — MUST pass before you trust the run

P0.2 fuses the ZJU `-SNR{n}-` ordinal steps into one session by stripping the ordinal
in `session_stem`. But the session union key is
**`(tax_key, condition, stem)`** (`build_manifests.py`, `build_sessions`, ~line 858).
If `condition` varies *across* the SNR1/2/3 steps of one ZJU class, the stem-union
will **not** fuse them and P0.2 is a no-op for that class.

**Confirm `condition` is constant across the ordinal steps** before believing the
81→27 collapse. After building `recordings.csv` (step 3 below), for the ZJU root:

```powershell
# group by (tax_key, stem-after-SNR-strip); condition must be single-valued per group
python - <<'PY'
import csv, collections, re, io
RE = re.compile(r"-SNR\d+(?=-|$)")
seen = collections.defaultdict(set)
with io.open(r"E:\Drone_dataset\RFUAV-v2\manifests\recordings.csv", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        stem = RE.sub("", (r.get("stem") or ""))
        seen[(r.get("tax_key",""), stem)].add(r.get("condition",""))
bad = {k: v for k, v in seen.items() if len(v) > 1}
print("groups where condition VARIES across SNR steps:", len(bad))
for k, v in list(bad.items())[:10]:
    print("  ", k, "->", sorted(v))
PY
```

- **0 groups vary** → precondition holds, the stem-union fuses the steps, proceed.
- **>0 vary** → STOP. P0.2 must additionally neutralize the `condition` axis for those
  classes before the merge is correct. Report the offending `(tax_key, stem)` groups;
  do not publish 81→27 until resolved.

---

## 2. E-016 before/after protocol (produces the numbers for ERRATA)

The `.pre-p0.bak` backups are the pristine pre-fix baseline (verified: they contain
**0** occurrences of `_RE_ZJU_SNR` and `use_sha`; the live files contain the edits).

```powershell
# BEFORE: build manifests from the pre-P0 build_manifests
copy tools\rfuav_v2\build_manifests.py            _live_build_manifests.py
copy tools\rfuav_v2\build_manifests.py.pre-p0.bak tools\rfuav_v2\build_manifests.py
python tools\rfuav_v2\build_manifests.py --manifests E:\Drone_dataset\RFUAV-v2\manifests_BEFORE `
    --inventory <e1_inventory.csv> --hashes <e1_hashes.csv>
copy _live_build_manifests.py tools\rfuav_v2\build_manifests.py   # restore the fixed version

# AFTER: build manifests from the fixed build_manifests (this is also step 3)
python tools\rfuav_v2\build_manifests.py --manifests E:\Drone_dataset\RFUAV-v2\manifests `
    --inventory <e1_inventory.csv> --hashes <e1_hashes.csv>
```

Then diff the ZJU session count between `manifests_BEFORE\sessions.csv` and
`manifests\sessions.csv`. Expected headline: **81 → 27** ZJU sessions, **21 classes**
degraded off `grouped_cv`.

The **`all_roots` 66.7 %→33.3 %** grouped_cv figure only appears when foreign roots are
kept in the split (see step 4b) — the ZJU root is dropped by default, so with the default
`make` you will *not* see it. The 81→27 session collapse, by contrast, is entirely in
`build_manifests` over the whole inventory and shows up regardless.

---

## 3. Build manifests (the fixed / "AFTER" run)

```powershell
python tools\rfuav_v2\build_manifests.py `
    --manifests E:\Drone_dataset\RFUAV-v2\manifests `
    --inventory <e1_inventory.csv> `
    --hashes    <e1_hashes.csv>            # col: path,size,sha512  -> enables P0.5 relation-3
# writes: recordings.csv, classes.csv, sessions.csv, leakage_groups.csv,
#         library_totals.csv, dedup_*.csv   (all under RFUAV-v2\manifests)
```

The E-016 diagnostic log (build_manifests.py:1262/1268) will print
`sessions = N (stem-only would be M; tail-of + sha512 merged K further, of which
sha512 alone merged J)` — record J for the P0.5 marginal-effect claim (gap 75:
expected ~1/25 hash groups, ~0.80 GB — report it that narrow, don't oversell).

---

## 4. Build the leakage-clean split

### 4a. Default (own-capture only — the publishable split)

```powershell
python tools\rfuav_v2\split_grouped.py make `
    --recordings E:\Drone_dataset\RFUAV-v2\manifests\recordings.csv `
    --out        E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv `
    --folds 3
# emits per-recording: session_id, taxonomy_id, split, source_relpath,
#   delivered_duration_s, is_slice, is_uncut, fold, eval_tier, ...
```

`make` **drops foreign roots (ZJU, .dat_pic) by default** — that is intended for the
headline split. `--own-capture-only` is accepted-and-ignored (already the default).

### 4b. `all_roots` variant (ONLY to reproduce the E-016 66.7→33.3 figure)

```powershell
python tools\rfuav_v2\split_grouped.py make `
    --recordings E:\Drone_dataset\RFUAV-v2\manifests\recordings.csv `
    --out        E:\Drone_dataset\RFUAV-v2\manifests\assignment_allroots.csv `
    --folds 3 --include-foreign-roots       # DANGER: pulls ZJU back in
```

Use this variant's `eval_tier` distribution to show `all_roots grouped_cv` fell from
66.7 % (pre-fix session count) to 33.3 % (post-fix) — i.e. "mixing roots looked better"
was a session-miscount artifact.

---

## 5. Carry the split down to the .jpg frames

```powershell
# dry-run first: no copies, just placement counts
python tools\rfuav_v2\split_images_grouped.py `
    --images     <root of image tree: <images>\<class>\*.jpg> `
    --split-file E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv `
    --out        E:\Drone_dataset\RFUAV-v2\split_images `
    --dry-run
# then the real copy (copy2, originals untouched):
python tools\rfuav_v2\split_images_grouped.py `
    --images     <image tree> `
    --split-file E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv `
    --out        E:\Drone_dataset\RFUAV-v2\split_images
```

Sanity: images whose recording isn't in the split, or whose basename collides across
splits, land in `unassigned/` (never train/valid). A non-trivial `unassigned` or
`collision` count is a signal to investigate, not to ignore.

---

## 6. Measure NAIVE (byte-identity) leakage — dupsplit2

**Gotcha: `dupsplit2.py` has NO CLI.** It hardcodes (dupsplit2.py:47-49):

```
M    = E:\Drone_dataset\RFUAV-v2\manifests
ROOT = E:\Drone_dataset\RFUAV\            # <-- prefix for the real .jpg files it hashes
OUT  = M\dupsplit_full.txt
```

It reads `M\e1_inventory.csv`, groups frames into trees/splits by path
(`train|valid|val|test|tests|eval`), SHA-256s every `ROOT + relpath`, and reports
`ident%` = held-out frames whose bytes match a train frame, per tree.

Because the new split was **copied into `RFUAV-v2\split_images\`** (not the original
`RFUAV\` tree), you must, **for this measurement only**, repoint dupsplit2 at the new
tree — back it up first (gitignored file, no git rollback):

```powershell
copy tools\rfuav_v2\dupsplit2.py tools\rfuav_v2\dupsplit2.py.pre-p0.bak
# edit dupsplit2.py:48  ->  ROOT = "E:\\Drone_dataset\\RFUAV-v2\\split_images\\"
# place an inventory of that new tree at  M\e1_inventory.csv
#   (dupsplit2 format: header with `path` (or source_relpath) + `size`/`size_bytes`,
#    relpaths carrying the split segment, root_marker "RFUAV-v2\\split_images\\")
python tools\rfuav_v2\dupsplit2.py     # writes M\dupsplit_full.txt
copy tools\rfuav_v2\dupsplit2.py.pre-p0.bak tools\rfuav_v2\dupsplit2.py   # restore
```

**Success criterion:** cross-split `ident%` → **~0** (was E-013 = 100 %, 12080/12080).

> CSV-only pre-flight (no real files, runs anywhere): `leakage_report.py
> --image-inventory <inv>` prints which trees dupsplit2 *would* hash and which it would
> skip. It deliberately never prints an ident% of its own — that number is dupsplit2's.

---

## 7. Confirm GROUPED leakage is zero — split_grouped check

```powershell
python tools\rfuav_v2\split_grouped.py check `
    --assignment E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv `
    --evidence
```

**Success criterion:** `0` sessions straddle a split boundary
(the old leaky split had `102/102 recordings in ≥2 splits`).

---

## 8. The honesty report — leakage_report (all four sections together)

```powershell
python tools\rfuav_v2\leakage_report.py `
    --assignment     E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv `
    --image-inventory M\e1_inventory.csv          # optional; adds the (1) pre-flight
# exit 0 = no grouped straddle; exit 1 = grouped leakage detected
```

Prints, and refuses to omit: (1) byte-identity note + pre-flight, (2) grouped leakage,
(3) files/session (the gap-58 singleton fraction), (4) per-class `eval_tier` +
`tier_basis` + quotable. **Publish all four together.** A clean (2) with a high
singleton fraction in (3) or many `NO` rows in (4) is *leakage-clean but thin*, not
*validated*. **Do not quote any accuracy without this block beside it** — and accuracy
itself is out of scope for this batch (needs a training run, downstream).

---

## 9. Success criteria (the whole run passed iff all hold)

| Check | Tool | Was | Must be |
|---|---|---|---|
| Byte-identity cross-split | `dupsplit2` | 100 % (12080/12080) | **~0 %** |
| Grouped straddle | `split_grouped check` | 102/102 recs in ≥2 splits | **0** |
| ZJU sessions | `sessions.csv` before/after | 81 | **27** (21 classes degraded) |
| `leakage_report` | exit code | — | **0**, all 4 sections printed |
| Precondition | §1 | — | **0 groups** where condition varies |

---

## 10. Rollback

- **Everything scale-run is under `E:\Drone_dataset\RFUAV-v2\`** → delete that dir to
  undo the run (manifests, assignment CSVs, `split_images\` copies). Originals in
  `RFUAV\` were never written.
- **Gitignored source files** (no git history): restore from backups —
  `build_manifests.py.pre-p0.bak`, `dialect.py.pre-p0.bak`, `split_grouped.py.pre-p0.bak`,
  and `dupsplit2.py.pre-p0.bak` (only if you edited ROOT in step 6).
- **Git-tracked source:** `utils/preprocessor.py` (shows `M`) →
  `git checkout -- utils/preprocessor.py`.

---

## 11. Downstream (NOT this batch — planned, not executed here)

- Naive-vs-grouped **accuracy** side-by-side (requires a training run on the box).
- The `same_day_only` tier (gap 102).
- Publish E-016 in `ERRATA.md` once §2 confirms 81→27 / 66.7→33.3 on real data.
- The AVATA2 `SNR{n}dB` native ladder is the *same* leakage pattern (attenuation steps
  of one emission). It is deliberately **preserved** by the regex (`-SNR\d+(?=-|$)`, not
  `(?!dB)`) because it is a goal-3a calibration asset; the leakage-correct treatment is
  one session with the ladder as a stratification/condition column — **measure it on the
  box as a parallel case before applying**, do not blind-edit it on Mac.
