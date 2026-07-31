# -*- coding: utf-8 -*-
"""Hypothesis test: the 88 byte-refuted pairs are INTERIOR windows, not tails.
A tail search finds suffixes only; an interior window [A,B) of a longer source
is structurally invisible to it.  Test: take the refuted FPV COMBO 10m packs,
find every plausible source in the library, and byte-probe at the CLAIMED
offset A*rate*bps (and A-1, given the E-012 off-by-one pattern).
READ-ONLY: opens sources with 'rb', seeks, reads 64 KB windows."""
import csv, hashlib, io, os

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
PREFIX = "E:\\Drone_dataset\\RFUAV\\"
W = 65536
BPS = 8
RATE = 100000000.0

inv = {}
with io.open(os.path.join(M, "e1_inventory.csv"), "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        inv[row["path"]] = int(row["size"])

# every FPV COMBO signal file in the library, with size
cands = [(p, s) for p, s in inv.items()
         if ("FPVCOMBO" in p.upper() or "FPV COMBO" in p.upper())
         and p.lower().endswith((".iq", ".dat"))]
cands.sort(key=lambda t: -t[1])
print("FPV COMBO signal files in library: %d" % len(cands))
for p, s in cands[:14]:
    print("  %14d B  %8.3f s  %s" % (s, s / (RATE * BPS), p[len(PREFIX):]))

targets = [p for p, s in cands
           if "PublicDataset" in p and "VTSBW=10" in p and "pack1_1" in p]
targets.sort()
print("\ntargets (claimed 10-13s, refuted against the 2s crop): %d" % len(targets))


def win(path, off, n=W):
    with io.open(path, "rb") as f:
        f.seek(off)
        return f.read(n)


def h(b):
    return hashlib.sha256(b).hexdigest()[:12]


for t in targets[:4]:
    tsz = inv[t]
    base = t.split("\\")[-1]
    # claimed window from _A-Bs
    stem = base.rsplit(".", 1)[0]
    a = stem.split("_")[-1].replace("s", "").split("-")[0]
    try:
        A = float(a)
    except ValueError:
        continue
    print("\n=== %s  (%d B = %.3f s, claims start %ss) ===" % (base, tsz, tsz / (RATE * BPS), A))
    hit = False
    for src, ssz in cands:
        if src == t or ssz <= tsz:
            continue
        for cand_A in (A, A - 1.0):
            off = int(round(cand_A * RATE * BPS))
            if off + tsz > ssz:
                continue
            try:
                ok = 0
                for rel in (0, tsz // 2, max(0, tsz - W)):
                    if h(win(src, off + rel)) == h(win(t, rel)):
                        ok += 1
                if ok == 3:
                    print("  *** INTERIOR MATCH at start=%.0fs (claimed %.0f) %s" % (
                        cand_A, A, "EXACT" if cand_A == A else "OFF-BY-ONE"))
                    print("      source: %s (%d B)" % (src[len(PREFIX):], ssz))
                    hit = True
                    break
            except Exception as e:
                print("  err %s: %s" % (src[len(PREFIX):], e))
        if hit:
            break
    if not hit:
        print("  no interior match found among %d candidates" % len(cands))
