# -*- coding: utf-8 -*-
"""E-013 open item: is the blockwise released split recording-ALIGNED, or does
it cut INSIDE recordings?

Established already: the split is contiguous blocks, not per-frame random
(adjacent-index flip rate 0.4-2.2% measured, vs 2p(1-p) ~= 32% expected for a
per-frame random 80/20).  But blockwise != per-recording.  A block can be

  (a) a whole recording            -> grouped split, no leakage
  (b) an arbitrary chunk of one    -> the cut is inside a recording, leakage

The discriminator is the BLOCK COUNT vs the number of independent recordings
for that class:

  blocks <= recordings  -> cuts MAY fall on recording boundaries (need names)
  blocks >  recordings  -> at least one cut is provably inside a recording

This script only lists filenames (no file reads, no hashing) so it does not
contend with the byte-identity scan.  Recording counts come from
draft_sessions.csv / draft_recordings.csv, joined client-side.
"""
import collections
import os
import re
import sys

ROOT = r"E:\Drone_dataset\RFUAV"
TRUE = ("train", "valid", "val", "test", "tests", "eval")
IMG = (".jpg", ".jpeg", ".png", ".bmp")
NUMTAIL = re.compile(r"^(.*?)[ _\-]*\(?(\d+)\)?$")

# trees whose frames kept a per-class numeric counter
WANT = "imageset-alldrones-matlabpipeline"

# tree -> class -> list of (n, split, filename)
seen = collections.defaultdict(lambda: collections.defaultdict(list))
nonum = collections.Counter()

for dirpath, dirnames, filenames in os.walk(ROOT):
    low = dirpath.lower()
    if WANT not in low:
        continue
    parts = dirpath.split("\\")
    lowp = [p.lower() for p in parts]
    hit = [i for i, s in enumerate(lowp) if s in TRUE]
    if not hit:
        continue
    i = hit[-1]
    split = lowp[i]
    tree = "\\".join(parts[:i])
    cls = parts[-1] if i < len(parts) - 1 else "(flat)"
    for fn in filenames:
        if not fn.lower().endswith(IMG):
            continue
        stem = os.path.splitext(fn)[0]
        m = NUMTAIL.match(stem)
        if not m:
            nonum[tree] += 1
            continue
        seen[tree][cls].append((int(m.group(2)), split, fn))

print("TREES FOUND: %d" % len(seen))
for tree in sorted(seen):
    print("\n" + "=" * 78)
    print("TREE  %s" % tree)
    print("      frames with no numeric tail: %d" % nonum[tree])
    print("=" * 78)
    rows = []
    for cls in sorted(seen[tree]):
        items = sorted(seen[tree][cls])
        ns = [n for n, _, _ in items]
        per = collections.Counter(s for _, s, _ in items)
        # contiguous runs over the sorted index order
        runs = []
        for n, s, fn in items:
            if runs and runs[-1][0] == s:
                runs[-1][2] = n
                runs[-1][3] += 1
            else:
                runs.append([s, n, n, 1])
        flips = len(runs) - 1
        gaps = (max(ns) - min(ns) + 1) - len(set(ns))
        rows.append((cls, len(items), per, len(runs), flips, min(ns), max(ns),
                     gaps, len(ns) - len(set(ns)), runs))

    print("\n%-26s %6s %5s %6s %8s %s" %
          ("CLASS", "n", "blks", "range", "missing", "per-split"))
    for (cls, n, per, nb, fl, lo, hi, gaps, dups, runs) in rows:
        ps = " ".join("%s=%d" % (k, v) for k, v in sorted(per.items()))
        print("%-26s %6d %5d %6s %8d %s" %
              (cls[:26], n, nb, "%d-%d" % (lo, hi), gaps, ps))
        if dups:
            print("%-26s   !! duplicate indices: %d" % ("", dups))

    print("\n--- BLOCK STRUCTURE (split: first-last  xN) ---")
    for (cls, n, per, nb, fl, lo, hi, gaps, dups, runs) in rows:
        s = "  ".join("%s:%d-%d x%d" % (r[0], r[1], r[2], r[3])
                      for r in runs[:24])
        more = "  ... +%d more" % (len(runs) - 24) if len(runs) > 24 else ""
        print("  %-24s %s%s" % (cls[:24], s, more))

    tot = sum(r[1] for r in rows)
    tb = sum(r[3] for r in rows)
    print("\n  TOTAL frames %d   TOTAL blocks %d   classes %d" %
          (tot, tb, len(rows)))
