# -*- coding: utf-8 -*-
"""Quantify the leakage, and test the exp3 same-index collision for literal
duplicate files.

Three questions, each with a number attached:

  Q1 CONTAMINATION RATE.  For each released split tree, what fraction of TEST
     (or valid) frames belong to a pack that also appears in TRAIN?  This is the
     headline: a test frame whose recording is in train is not a test frame.
     Reported per tree in frames, not in packs, because frames are what the
     reported accuracy was averaged over.

  Q2 LITERAL DUPLICATES.  exp3 has train\\<cls>\\pack1(1).jpg AND
     valid\\<cls>\\pack1(1).jpg -- same pack, same window index.  Either the
     per-split numbering restarts (so they are different windows that collide in
     name) or they are the same rendered frame copied to both sides.  Size
     equality is weak; SHA-256 settles it.  Only the colliding pairs are read,
     a few MB, so this does not contend with E1.

  Q3 AUGMENTATION INHERITANCE.  dataset_aug mirrors dataset_or with an _AugM<k>
     suffix.  If the augmented tree was produced from the ALREADY-SPLIT original
     tree, then every augmented variant inherits the contamination rather than
     diluting it.  Testable from names: does each dataset_or\\<split>\\<cls>\\X.jpg
     have its dataset_aug\\<split>\\<cls>\\X.jpg_AugM*.jpg on the SAME side?
"""
import csv, io, os, re, collections, hashlib

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
ROOT = "E:\\Drone_dataset\\RFUAV\\"
TRUE = ("train", "valid", "val", "test", "tests", "eval")
PACK = re.compile(r"(pack\s*\d+)", re.I)

# tree -> split -> pack -> [relpaths]
T = collections.defaultdict(lambda: collections.defaultdict(
    lambda: collections.defaultdict(list)))
nopack = collections.Counter()

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
        tree, split = "\\".join(parts[:i]), low[i]
        base = os.path.splitext(parts[-1])[0]
        m = PACK.search(base)
        if not m:
            nopack[tree] += 1
            continue
        after = parts[i + 1:]
        cls = after[0] if len(after) >= 2 else "(flat)"
        # class lives in the filename for the YOLO-style images/ layout
        if cls.lower() == "images":
            cls = re.split(r"[_ ]*pack", base, flags=re.I)[0].strip() or "(?)"
        T[tree][split][(cls, m.group(1).lower().replace(" ", ""))].append(rel)

print("=== Q1 CONTAMINATION RATE (frames whose pack is also in train) ===\n")
print("  %-58s %7s %7s %8s" % ("tree \\ split", "frames", "leaked", "rate"))
grand_n = grand_l = 0
for tree in sorted(T):
    tr = set(T[tree].get("train", {}))
    if not tr:
        print("  %-58s (no train dir)" % tree[:58])
        continue
    print("  %s" % tree)
    for split in sorted(T[tree]):
        if split == "train":
            continue
        n = l = 0
        for k, paths in T[tree][split].items():
            n += len(paths)
            if k in tr:
                l += len(paths)
        if n:
            print("    %-56s %7d %7d %7.1f%%"
                  % (split, n, l, 100.0 * l / n))
            grand_n += n
            grand_l += l
    if nopack.get(tree):
        print("    %-56s %7d   (no pack id in name -> ungroupable)"
              % ("[frames excluded]", nopack[tree]))
print("\n  OVERALL: %d held-out frames, %d leaked = %.1f%%"
      % (grand_n, grand_l, 100.0 * grand_l / max(1, grand_n)))

print("\n=== Q2 exp3 same-index collisions: duplicate files or name reuse? ===\n")


def sha(p, cap=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


tree = "exp3"
tr, va = T[tree].get("train", {}), T[tree].get("valid", {})
pairs = 0
ident = 0
for k in sorted(set(tr) & set(va)):
    a = {os.path.basename(x): x for x in tr[k]}
    b = {os.path.basename(x): x for x in va[k]}
    for name in sorted(set(a) & set(b)):
        if pairs >= 12:
            break
        try:
            ha, hb = sha(ROOT + a[name]), sha(ROOT + b[name])
        except Exception as e:
            print("    ERR %s: %s" % (name, e))
            continue
        pairs += 1
        same = ha == hb
        ident += same
        print("    %-14s %-12s train=%s valid=%s  %s"
              % (k[0][:14], name[:12], ha, hb,
                 "IDENTICAL BYTES" if same else "different"))
print("\n    %d name-collision pairs checked, %d byte-identical" % (pairs, ident))

print("\n=== Q3 does augmentation inherit the split boundary? ===\n")
for pair in (("PublicDataset\\Dataset\\dataset_5Drones-raw+20dB_parural_usingAG"
              "\\dataset_or",
              "PublicDataset\\Dataset\\dataset_5Drones-raw+20dB_parural_usingAG"
              "\\dataset_aug"),):
    o, g = pair
    for split in ("train", "valid"):
        so = set()
        for k, ps in T[o].get(split, {}).items():
            for x in ps:
                so.add(os.path.basename(x))
        sg = set()
        for k, ps in T[g].get(split, {}).items():
            for x in ps:
                sg.add(re.sub(r"\.jpg_AugM\d+\.jpg$", ".jpg",
                              os.path.basename(x)))
        inter = len(so & sg)
        print("    %-6s  or=%6d  aug-stems=%6d  same-side overlap=%6d (%.1f%% of or)"
              % (split, len(so), len(sg), inter, 100.0 * inter / max(1, len(so))))
    # cross check: aug stem on the OPPOSITE side
    so_tr = set(os.path.basename(x) for k, ps in T[o].get("train", {}).items()
                for x in ps)
    sg_va = set(re.sub(r"\.jpg_AugM\d+\.jpg$", ".jpg", os.path.basename(x))
                for k, ps in T[g].get("valid", {}).items() for x in ps)
    print("    cross: train-original stems appearing as valid-augmented: %d"
          % len(so_tr & sg_va))
