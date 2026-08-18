"""Recon 2 for DroneRFa top-up:
(a) for every canonical DroneRFa recording, find the source reader
    (path + dtype) that reproduces slice_0000 (corr > 0.999 on 200k samples);
(b) cross-correlate slice_0000 between recordings of the same taxonomy+rf
    chain to detect format-twin duplicates (same capture, different encoding);
(c) print manifest split for each so straddling duplicates are visible.
"""
import os, json, glob, csv
import numpy as np

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
RFA_ROOT = r"E:\Drone_dataset\RFA"
MANIFEST = r"E:\Drone_dataset\RFUAV-v2\manifests\canonical_manifest_v1.csv"
NCHK = 200_000

split_of = {}
for r in csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")):
    split_of[(r["taxonomy_id"], r["recording"])] = r["split"]

index = {}
for dirpath, dirs, files in os.walk(RFA_ROOT):
    for fn in files:
        if fn.lower().endswith((".dat", ".bin", ".iq")):
            stem = os.path.splitext(fn)[0]
            index.setdefault(stem, []).append(os.path.join(dirpath, fn))


def read_as(path, kind, n):
    if kind == "c64":
        return np.fromfile(path, dtype=np.complex64, count=n).astype(np.complex128)
    dt = {"f32": np.float32, "i16": np.int16, "i32": np.int32}[kind]
    raw = np.fromfile(path, dtype=dt, count=2 * n).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def corr(a, b):
    n = min(len(a), len(b))
    if n < 1000:
        return 0.0
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) / d) if d > 0 else 0.0


recs = []
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        mpath = os.path.join(tdir, rec, "metadata.json")
        if not os.path.exists(mpath):
            continue
        md = json.load(open(mpath, encoding="utf-8"))
        if md.get("source_dataset") != "DroneRFa":
            continue
        s0 = os.path.join(tdir, rec, "slice_0000.iq")
        raw = np.fromfile(s0, dtype=np.float32, count=2 * NCHK).astype(np.float64)
        a = raw[0::2] + 1j * raw[1::2]
        recs.append({"tax": tax, "rec": rec, "md": md, "a": a,
                     "rf": md.get("rf_chain", rec[-1]),
                     "split": split_of.get((tax, rec), "?")})

print("=== (a) source reader match ===")
readers = {}
for r in recs:
    stem = r["md"].get("original_file", "")
    best = None
    for cand in index.get(stem, []):
        for kind in ("c64", "f32", "i16", "i32"):
            try:
                b = read_as(cand, kind, NCHK)
            except Exception:
                continue
            c = corr(r["a"], b)
            if best is None or c > best[2]:
                best = (cand, kind, c)
    if best:
        readers[(r["tax"], r["rec"])] = best
        tag = "OK" if best[2] > 0.999 else "WEAK"
        sub = os.path.relpath(best[0], RFA_ROOT)
        print("%-4s %-24s %-38s %-4s c=%.4f  %s"
              % (tag, r["tax"][:24], r["rec"][:38], best[1], best[2], sub))
    else:
        print("MISS %-24s %-38s no candidate" % (r["tax"][:24], r["rec"][:38]))

print()
print("=== (b) sibling duplicate check (same tax+rf, corr of slice_0000) ===")
from collections import defaultdict
groups = defaultdict(list)
for r in recs:
    groups[(r["tax"], str(r["rf"]))].append(r)
n_dup = n_straddle = 0
for key, members in sorted(groups.items()):
    if len(members) < 2:
        continue
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            c = corr(members[i]["a"], members[j]["a"])
            dup = c > 0.99
            straddle = dup and members[i]["split"] != members[j]["split"]
            if dup:
                n_dup += 1
            if straddle:
                n_straddle += 1
            flag = ("DUP+STRADDLE!" if straddle else ("DUP same-side" if dup else "distinct"))
            print("%-14s %-38s(%s) vs %-38s(%s) c=%.4f %s"
                  % (key[0][:14], members[i]["rec"][:38], members[i]["split"],
                     members[j]["rec"][:38], members[j]["split"], c, flag))
print("dup_pairs=%d straddling=%d" % (n_dup, n_straddle))
