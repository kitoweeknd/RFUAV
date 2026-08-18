"""Probe: can RFUAV canonical slices be reproduced from their source .iq at
the contiguous-from-zero offsets? Prints metadata fields + reader check for a
sample of recordings across source roots."""
import os, json, glob, random
import numpy as np

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
ROOTS = [r"E:\Drone_dataset\RFUAV", r"E:\Drone_dataset"]
NCHK = 200_000

recs = []
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        mp = os.path.join(tdir, rec, "metadata.json")
        if not os.path.exists(mp):
            continue
        md = json.load(open(mp, encoding="utf-8"))
        if md.get("source_dataset", "RFUAV") == "RFUAV":
            recs.append((tax, rec, md))

print("RFUAV recordings: %d" % len(recs))
md0 = recs[0][2]
print("metadata keys:", sorted(md0.keys()))

random.seed(7)
sample = random.sample(recs, 6)


def read_f32(path, start, n):
    raw = np.fromfile(path, dtype=np.float32, count=2 * n, offset=start * 8)
    raw = raw.astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) / d) if d > 0 else 0.0


for tax, rec, md in sample:
    rdir = os.path.join(CANON, tax, rec)
    slices = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))
    ss = os.path.getsize(slices[0]) // 8
    rel = md.get("source_relpath", "")
    src = None
    for root in ROOTS:
        p = os.path.normpath(os.path.join(root, rel))
        if rel and os.path.isfile(p):
            src = p
            break
    print("--- %s/%s" % (tax, rec))
    print("    ss=%d fs=%s relpath=%r -> %s" % (ss, md.get("sample_rate_hz"),
          rel[:70], "FOUND" if src else "NOT FOUND"))
    if not src:
        continue
    sz = os.path.getsize(src)
    print("    src=%.2f GB max_slices=%d" % (sz / 1e9, sz // (ss * 8)))
    a0 = read_f32(slices[0], 0, NCHK)
    b0 = read_f32(src, 0, NCHK)
    c0 = corr(a0, b0)
    a1 = read_f32(slices[1], 0, NCHK) if len(slices) > 1 else None
    c1 = corr(a1, read_f32(src, ss, NCHK)) if a1 is not None else -1
    alpha = complex(np.vdot(b0, a0) / np.vdot(b0, b0)) if c0 > 0.99 else 0
    print("    corr(slice0@0)=%.5f corr(slice1@ss)=%.5f alpha=%s" % (c0, c1, alpha))
