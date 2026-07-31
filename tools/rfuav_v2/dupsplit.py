# -*- coding: utf-8 -*-
"""Provenance-free leakage test: is a held-out frame the SAME FILE as a training
frame?

Why this test is better than the pack-id test it follows.  The pack-id test could
only speak for frames whose name kept a pack token; it had to exclude 17,921
frames that are named ``<CLASS><N>.jpg`` or ``(0).jpg``.  Byte identity needs no
naming convention at all, so it covers every frame -- and it is not an inference
about grouping, it is the strongest possible statement: the model was evaluated
on a file it had already trained on.

Bounded cost: only the split trees are touched, smallest first, and any tree
whose train+held-out payload exceeds CAP_GB is reported and skipped rather than
read.  Hashing is on a 1 MiB stride so a 100 KB JPEG costs one read.
"""
import csv, io, os, re, sys, collections, hashlib

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
ROOT = "E:\\Drone_dataset\\RFUAV\\"
TRUE = ("train", "valid", "val", "test", "tests", "eval")
CAP_GB = 30.0

T = collections.defaultdict(lambda: collections.defaultdict(list))  # tree->split->[(rel,size)]
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
        T["\\".join(parts[:i])][low[i]].append((rel, int(row["size"])))

print("=== split trees, by payload ===")
order = []
for tree in T:
    tot = sum(sz for sp in T[tree] for _, sz in T[tree][sp])
    n = sum(len(T[tree][sp]) for sp in T[tree])
    order.append((tot, n, tree))
order.sort()
for tot, n, tree in order:
    print("  %8.2f GB %8d frames  %s   splits=%s"
          % (tot / 1e9, n, tree[:60], sorted(T[tree])))

print("\n=== exact cross-split duplicates (SHA-256, full file) ===")


def sha(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.digest()


for tot, n, tree in order:
    if "train" not in T[tree]:
        print("\n  %s -> no train split, skipped" % tree[:70])
        continue
    if tot / 1e9 > CAP_GB:
        print("\n  %s -> %.1f GB > cap, skipped (not read)" % (tree[:70], tot / 1e9))
        continue
    print("\n  %s  (%.2f GB)" % (tree, tot / 1e9))
    trh = {}
    for rel, sz in T[tree]["train"]:
        d = sha(ROOT + rel)
        if d:
            trh.setdefault(d, rel)
    sys.stdout.write("    train frames hashed: %d (distinct %d, %d internal dups)\n"
                     % (len(T[tree]["train"]), len(trh),
                        len(T[tree]["train"]) - len(trh)))
    for split in sorted(T[tree]):
        if split == "train":
            continue
        nn = dup = 0
        ex = []
        for rel, sz in T[tree][split]:
            d = sha(ROOT + rel)
            if d is None:
                continue
            nn += 1
            if d in trh:
                dup += 1
                if len(ex) < 3:
                    ex.append((rel, trh[d]))
        print("    %-6s frames=%6d  byte-identical to a TRAIN frame=%6d  %5.1f%%"
              % (split, nn, dup, 100.0 * dup / max(1, nn)))
        for a, b in ex:
            print("        %s\n          == %s" % (a, b))
