# -*- coding: utf-8 -*-
"""E1 progress + resolve the 25 dtype_unprobed files.

Discriminator established in E-009b: read the file as float32 and measure the
finite-value fraction.  A true cf32_le file is 100.00% finite with rms in
1e-4..1e0; interleaved int16 misread as float32 is 91-99% finite with rms ~1e37
and min approaching -FLT_MAX (-3.4e38), because int16 pairs alias onto
denormals/NaN/huge exponents.  Cheap: only the head of each file is read.
READ-ONLY.
"""
import csv, io, os, sys, time

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
PREFIX = "E:\\Drone_dataset\\RFUAV\\"

# ---- E1 progress ----------------------------------------------------------
n = b = 0
with io.open(M + r"\e1_hashes.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            try:
                b += int(row[1]); n += 1
            except ValueError:
                pass
tot = tn = 0
SIG = (".iq", ".dat", ".bin", ".mat")
with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["path"].lower().endswith(SIG):
            tot += int(row["size"]); tn += 1
print("=== E1 SHA-512 ===")
print("  hashed  : %d/%d files   %.1f / %.1f GB   %.1f%%"
      % (n, tn, b / 2.0**30, tot / 2.0**30, 100.0 * b / tot))
print("  updated : %s" % time.ctime(os.path.getmtime(M + r"\e1_hashes.csv")))

# ---- the 25 unprobed dtypes ----------------------------------------------
targets = []
with io.open(M + r"\draft_anomalies.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row["kind"] == "dtype_unprobed":
            targets.append(row["relpath"])
print("\n=== dtype_unprobed = %d files ===" % len(targets))

try:
    import numpy as np
except ImportError:
    print("  numpy unavailable -- cannot run the float32 discriminator here")
    sys.exit(0)

NREAD = 8 * 1024 * 1024   # 8 MB head is plenty for the statistic


def probe(path):
    with io.open(path, "rb") as fh:
        raw = fh.read(NREAD)
    nb = (len(raw) // 4) * 4
    if nb == 0:
        return None
    a = np.frombuffer(raw[:nb], dtype="<f4")
    fin = np.isfinite(a)
    frac = 100.0 * fin.sum() / a.size
    g = a[fin]
    rms = float(np.sqrt(np.mean(g.astype(np.float64) ** 2))) if g.size else float("nan")
    return frac, rms, float(g.min()) if g.size else float("nan"), \
        float(g.max()) if g.size else float("nan"), a.size


print("  %-64s %8s %10s %12s %8s" % ("file", "finite%", "rms", "min", "verdict"))
verdict_count = {}
for rel in sorted(targets):
    p = PREFIX + rel
    try:
        r = probe(p)
    except (IOError, OSError) as e:
        print("  %-64s  ERROR %s" % (rel[-64:], e.__class__.__name__))
        continue
    if r is None:
        print("  %-64s  EMPTY" % rel[-64:])
        continue
    frac, rms, mn, mx, ns = r
    # decision rule from E-009b
    if frac >= 99.999 and rms < 1e3:
        v = "cf32_le"
    elif frac < 99.99 or rms > 1e10 or mn < -1e30:
        v = "ci16_le"
    else:
        v = "AMBIGUOUS"
    verdict_count[v] = verdict_count.get(v, 0) + 1
    print("  %-64s %7.3f%% %10.2e %12.2e %8s" % (rel[-64:], frac, rms, mn, v))

print("\n  verdicts: %s" % verdict_count)
