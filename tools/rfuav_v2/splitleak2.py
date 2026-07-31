# -*- coding: utf-8 -*-
"""splitleak v2.  v1 had two defects that made its headline worthless:

  (a) ``for k, v in sorted(multi)`` iterates a dict's KEYS, and each key here is
      the tuple ``(cls, pack)`` -- so it unpacked cls into k and pack into v and
      then char-sorted the pack string.  The len(splits)>1 test itself was fine;
      the report of WHICH splits was never printed at all.
  (b) the split vocabulary mixed ``train/valid/test`` with ``close/middle/far``.
      Those are not the same kind of thing: close/middle/far are MEASUREMENT
      CONDITIONS (distance / SNR), so one pack legitimately appears in several.
      Counting that as leakage would be a false positive.

v2 keeps the two families strictly apart and prints the evidence.

It also answers the question v1's P2 could not.  A flip rate of 1.0-2.2% is far
BELOW the ~32% expected of a per-frame random 80/20 split, so the split is
blockwise, not per-frame.  But blockwise is ambiguous: blocks could be whole
recordings (correct, grouped) or arbitrary contiguous chunks of ONE recording
(leaky).  The discriminator is the BLOCK COUNT versus the number of independent
source recordings for that class: if a class has 12 blocks and 12 recordings,
the split plausibly respects recordings; if it has 12 blocks and 2 recordings,
the cut is inside recordings and train/valid are adjacent windows of the same
capture.
"""
import csv, io, os, collections, re

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
TRUE = ("train", "valid", "val", "test", "tests", "eval")
COND = ("close", "middle", "far")

PACK = re.compile(r"(pack\s*\d+)", re.I)
NUM = re.compile(r"^(.*?)[ _(]*(\d+)\)?$")

pack_splits = collections.defaultdict(lambda: collections.defaultdict(set))
pack_ex = {}
idx = collections.defaultdict(lambda: collections.defaultdict(dict))
seen_layout = collections.Counter()

with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        p = row["path"]
        if not p.lower().endswith(".jpg"):
            continue
        rel = p.split("Drone_dataset\\RFUAV\\", 1)[-1]
        parts = rel.split("\\")
        low = [s.lower() for s in parts]
        hit = [i for i, s in enumerate(low) if s in TRUE]
        if not hit:
            continue
        i = hit[-1]
        split = low[i]
        tree = "\\".join(parts[:i])
        cond = "|".join(s for s in low[:i] if s in COND) or "-"
        after = parts[i + 1:]
        cls = after[0] if len(after) >= 2 else "(flat)"
        base = os.path.splitext(parts[-1])[0]
        seen_layout[(tree, cond, len(after))] += 1

        m = PACK.search(base)
        if m:
            key = (cls, m.group(1).lower().replace(" ", ""), cond)
            pack_splits[tree][key].add(split)
            pack_ex.setdefault((tree, key, split), rel)

        m2 = NUM.match(base)
        if m2:
            stem = m2.group(1).strip(" _-").upper().replace(" ", "").replace("_", "")
            if stem == cls.upper().replace(" ", "").replace("_", ""):
                idx[tree][cls][int(m2.group(2))] = split

print("=== P1 (fixed): same (class, pack, condition) in TWO TRUE SPLITS ===")
print("    condition held CONSTANT, so close/middle/far cannot cause a hit.\n")
leak = 0
for tree in sorted(pack_splits):
    keys = pack_splits[tree]
    multi = [(k, v) for k, v in keys.items() if len(v) > 1]
    print("  tree: %s" % tree)
    print("    (class,pack,cond) groups: %d   spanning >1 TRUE split: %d"
          % (len(keys), len(multi)))
    for k, v in sorted(multi, key=lambda t: str(t[0]))[:8]:
        leak += 1
        print("      LEAK  cls=%-18s pack=%-8s cond=%-8s splits=%s"
              % (k[0][:18], k[1], k[2], sorted(v)))
        for s in sorted(v):
            ex = pack_ex.get((tree, k, s))
            if ex:
                print("              %-6s <- %s" % (s, ex))
    print("")
print("  P1 verdict: %d (class,pack,cond) groups straddle a released split"
      % leak)

print("\n=== P2: block structure vs. expectation ===")
print("  a random 80/20 per-frame split flips side ~32%% of the time.")
print("  %-46s %-15s %6s %7s %7s" % ("tree", "class", "n", "flips%", "blocks"))
tot = collections.Counter()
for tree in sorted(idx):
    for cls in sorted(idx[tree]):
        d = idx[tree][cls]
        if len(d) < 20 or len(set(d.values())) < 2:
            continue
        ks = sorted(d)
        flips = sum(1 for a, b in zip(ks, ks[1:]) if d[a] != d[b])
        blocks = flips + 1
        tot[(tree, cls)] = blocks
        print("  %-46s %-15s %6d %6.1f%% %7d"
              % (tree[:46], cls[:15], len(ks), 100.0 * flips / (len(ks) - 1),
                 blocks))
if tot:
    v = list(tot.values())
    print("\n  blocks per (tree,class): min=%d median=%d max=%d over %d classes"
          % (min(v), sorted(v)[len(v) // 2], max(v), len(v)))
    print("  -> compare against recordings-per-class from the manifest; blocks")
    print("     >> recordings means the cut is INSIDE recordings.")
