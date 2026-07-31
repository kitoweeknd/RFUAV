# -*- coding: utf-8 -*-
"""Does the RELEASED, on-disk train/valid split put the same recording on both
sides?  Answer from paths alone -- no model, no training run.

Two independent probes, because the frame trees differ in how much provenance
they kept:

  P1  pics_exp1_alldrones keeps the pack id in the filename
      (``DAUTEL_pack1 (1).jpg``).  So ask directly: does any (class, pack)
      appear in more than one split directory?  If yes, that is leakage
      demonstrated at the path level -- the same 1-second pack, which E-011
      already proved is a re-slice of one rawdata recording, is in train and in
      test.  Nothing about this depends on my interpretation.

  P2  the augmentation trees kept only ``<CLASS><N>.jpg``, a per-class running
      counter.  Provenance is gone, but the counter still leaks the SPLIT
      POLICY: if train and valid hold interleaved indices, the split was drawn
      per-frame at random, which is the worst case (adjacent windows of one
      recording land on both sides).  Contiguous blocks would at least mean a
      temporal cut.  Measured by counting how often consecutive integers change
      side.

Inventory-only: reads no image, touches no disk beyond the CSV.
"""
import csv, io, os, collections, re

M = r"E:\Drone_dataset\RFUAV-v2\manifests"

# split-bearing path components, as they actually appear on disk
SPLITS = ("train", "valid", "val", "test", "close", "middle", "far")

trees = collections.defaultdict(lambda: collections.defaultdict(set))   # tree -> (cls,pack) -> splits
idx = collections.defaultdict(lambda: collections.defaultdict(dict))    # tree -> cls -> n -> split
tree_of = {}

PACK = re.compile(r"(pack\d+)", re.I)
NUM = re.compile(r"^(.*?)[ _(]*(\d+)\)?$")

with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        p = row["path"]
        if not p.lower().endswith(".jpg"):
            continue
        rel = p.split("Drone_dataset\\RFUAV\\", 1)[-1]
        parts = rel.split("\\")
        sp = None
        for i, seg in enumerate(parts):
            if seg.lower() in SPLITS:
                sp = (i, seg.lower())
        if sp is None:
            continue
        i, split = sp
        tree = "\\".join(parts[:i])
        if i + 1 >= len(parts):
            continue
        cls = parts[i + 1] if i + 2 <= len(parts) - 1 else "(flat)"
        base = os.path.splitext(parts[-1])[0]

        m = PACK.search(base)
        if m:
            trees[tree][(cls, m.group(1).lower())].add(split)

        m2 = NUM.match(base)
        if m2 and m2.group(1).strip(" _-").upper().replace(" ", "") == \
                cls.upper().replace(" ", "").replace("_", ""):
            try:
                idx[tree][cls][int(m2.group(2))] = split
            except ValueError:
                pass

print("=== P1: same (class, pack) on BOTH sides of a released split ===")
any_leak = False
for tree in sorted(trees):
    keys = trees[tree]
    multi = {k: v for k, v in keys.items() if len(v) > 1}
    if not keys:
        continue
    print("\n  tree: %s" % tree)
    print("    (class,pack) groups: %d   spanning >1 split: %d"
          % (len(keys), len(multi)))
    if multi:
        any_leak = True
        for k, v in sorted(multi)[:12]:
            print("      LEAK  %-34s in %s" % (str(k), sorted(v)))
print("\n  P1 verdict: %s" % ("LEAKAGE DEMONSTRATED at path level"
                              if any_leak else "no pack spans two splits"))

print("\n=== P2: split policy inferred from the per-class running counter ===")
print("  %-52s %-14s %6s %8s %9s" % ("tree", "class", "n", "sides", "flips"))
for tree in sorted(idx):
    for cls in sorted(idx[tree]):
        d = idx[tree][cls]
        if len(d) < 20:
            continue
        ks = sorted(d)
        sides = sorted(set(d.values()))
        if len(sides) < 2:
            continue
        flips = sum(1 for a, b in zip(ks, ks[1:]) if d[a] != d[b])
        print("  %-52s %-14s %6d %8s %8.1f%%"
              % (tree[:52], cls[:14], len(ks), "/".join(sides),
                 100.0 * flips / max(1, len(ks) - 1)))
print("\n  interpretation: flips near 0%% = contiguous blocks (temporal cut);")
print("  flips at the class-balance rate = per-frame RANDOM split (worst case).")
