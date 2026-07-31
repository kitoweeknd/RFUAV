# -*- coding: utf-8 -*-
"""E-013 closer: the released blockwise split cuts INSIDE each group.

blockcount.py showed every class has an EVEN number of contiguous blocks in a
strict train,valid,train,valid... alternation.  If blocks were whole recordings
assigned to sides, the train fraction of each (train,valid) cycle would vary
with the recordings' lengths.  If instead each group is cut at a fixed
percentile, every cycle shows the SAME fraction regardless of its length.

This measures that fraction per cycle.  A tight distribution across cycles of
widely differing length is proof the cut is computed inside the group.
"""
import collections
import os
import re

ROOT = r"E:\Drone_dataset\RFUAV"
TRUE = ("train", "valid", "val", "test", "tests", "eval")
IMG = (".jpg", ".jpeg", ".png", ".bmp")
NUMTAIL = re.compile(r"^(.*?)[ _\-]*\(?(\d+)\)?$")
WANT = "imageset-alldrones-matlabpipeline"

seen = collections.defaultdict(list)
for dirpath, dirnames, filenames in os.walk(ROOT):
    if WANT not in dirpath.lower():
        continue
    parts = dirpath.split("\\")
    lowp = [p.lower() for p in parts]
    hit = [i for i, s in enumerate(lowp) if s in TRUE]
    if not hit:
        continue
    i = hit[-1]
    if i >= len(parts) - 1:
        continue
    for fn in filenames:
        if not fn.lower().endswith(IMG):
            continue
        m = NUMTAIL.match(os.path.splitext(fn)[0])
        if m:
            seen[parts[-1]].append((int(m.group(2)), lowp[i]))

allfrac = []
print("%-26s %6s %7s %8s   %s" % ("CLASS", "n", "groups", "pattern", "train-fraction per cycle"))
bad_pattern = []
for cls in sorted(seen):
    items = sorted(seen[cls])
    runs = []
    for n, s in items:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    seq = [r[0] for r in runs]
    ok = (len(runs) % 2 == 0 and
          all(seq[k] == ("train" if k % 2 == 0 else "valid")
              for k in range(len(seq))))
    if not ok:
        bad_pattern.append((cls, seq))
    fr = []
    for k in range(0, len(runs) - 1, 2):
        if runs[k][0] == "train" and runs[k + 1][0] == "valid":
            t, v = runs[k][1], runs[k + 1][1]
            fr.append(t / float(t + v))
            allfrac.append((t / float(t + v), t + v, cls))
    print("%-26s %6d %7d %8s   %s" % (
        cls[:26], len(items), len(runs) / 2.0,
        "OK" if ok else "IRREG",
        " ".join("%.3f" % x for x in fr[:14])))

n = len(allfrac)
xs = sorted(x[0] for x in allfrac)
mean = sum(xs) / n
var = sum((x - mean) ** 2 for x in xs) / n
print("\n=== TRAIN-FRACTION ACROSS ALL %d (train,valid) CYCLES ===" % n)
print("  mean   %.4f" % mean)
print("  stdev  %.4f" % (var ** 0.5))
print("  min    %.4f   max %.4f" % (xs[0], xs[-1]))
print("  median %.4f" % xs[n // 2])
sz = sorted(allfrac, key=lambda t: t[1])
print("\n  smallest group %d frames -> frac %.4f  (%s)" % (sz[0][1], sz[0][0], sz[0][2]))
print("  largest  group %d frames -> frac %.4f  (%s)" % (sz[-1][1], sz[-1][0], sz[-1][2]))
print("\n  cycles with frac in [0.29,0.31]: %d / %d" %
      (sum(1 for x in xs if 0.29 <= x <= 0.31), n))
print("  TOTAL groups (= cycles) across %d classes: %d" % (len(seen), n))
if bad_pattern:
    print("\n  IRREGULAR alternation: %d classes" % len(bad_pattern))
    for cls, seq in bad_pattern[:10]:
        print("    %-24s %s" % (cls[:24], ",".join(seq)))
else:
    print("\n  alternation strictly train,valid,... in ALL %d classes" % len(seen))
