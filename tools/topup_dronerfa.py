"""Top up every canonical DroneRFa recording toward the TARGET slice count.

For each recording the source reader (path + dtype) is re-derived by matching
slice_0000 against candidate files (corr > 0.9999 on 200k samples), and the
conversion scale alpha is measured by least squares on slice_0000 and
cross-checked on slice_0001. Appended slices use the exact importer formula
(slice si = source samples [si*ss, (si+1)*ss)) so numbering stays consistent.
Originals are READ-ONLY; writes only under RFUAV-v2\\canonical.
"""
import os, json, glob
import numpy as np

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
RFA_ROOT = r"E:\Drone_dataset\RFA"
TARGET = 20
NCHK = 200_000

BYTES_PER_SAMP = {"f32": 8, "i16": 4, "i32": 8}

index = {}
for dirpath, dirs, files in os.walk(RFA_ROOT):
    for fn in files:
        if fn.lower().endswith((".dat", ".bin", ".iq")):
            index.setdefault(os.path.splitext(fn)[0], []).append(
                os.path.join(dirpath, fn))


def read_seg(path, kind, start, n):
    """Read n complex samples starting at sample offset `start` as complex128."""
    if kind == "f32":  # float32 interleaved == complex64 on disk
        raw = np.fromfile(path, dtype=np.float32, count=2 * n, offset=start * 8)
    elif kind == "i16":
        raw = np.fromfile(path, dtype=np.int16, count=2 * n, offset=start * 4)
    else:
        raw = np.fromfile(path, dtype=np.int32, count=2 * n, offset=start * 8)
    raw = raw.astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def read_slice(path, n=None):
    raw = np.fromfile(path, dtype=np.float32, count=(2 * n if n else -1))
    raw = raw.astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) / d) if d > 0 else 0.0


def alpha_of(a, b):
    """least-squares alpha with  a ~= alpha * b"""
    n = min(len(a), len(b))
    return complex(np.vdot(b[:n], a[:n]) / np.vdot(b[:n], b[:n]))


done = skipped = failed = new_slices = 0
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        mpath = os.path.join(rdir, "metadata.json")
        if not os.path.exists(mpath):
            continue
        md = json.load(open(mpath, encoding="utf-8"))
        if md.get("source_dataset") != "DroneRFa":
            continue
        cur = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))
        n_cur = len(cur)
        ss = int(md.get("slice_samples") or 0)
        if not ss:
            print("FAIL %s/%s: no slice_samples in metadata" % (tax, rec))
            failed += 1
            continue
        if n_cur >= TARGET:
            skipped += 1
            continue

        # derive verified reader
        a0 = read_slice(cur[0], NCHK)
        best = None
        for cand in index.get(md.get("original_file", ""), []):
            for kind in ("f32", "i16", "i32"):
                try:
                    b = read_seg(cand, kind, 0, NCHK)
                except Exception:
                    continue
                c = corr(a0, b)
                if best is None or c > best[2]:
                    best = (cand, kind, c)
        if best is None or best[2] < 0.9999:
            print("FAIL %s/%s: no reader match (best=%s)" % (tax, rec, best and best[2]))
            failed += 1
            continue
        spath, kind, c0 = best

        # scale: measure on slice 0, cross-check on slice 1
        alpha0 = alpha_of(a0, read_seg(spath, kind, 0, NCHK))
        if n_cur > 1:
            a1 = read_slice(cur[1], NCHK)
            b1 = read_seg(spath, kind, ss, NCHK)
            c1 = corr(a1, b1)
            alpha1 = alpha_of(a1, b1)
            drift = abs(alpha1 / alpha0 - 1)
            if c1 < 0.9999 or drift > 1e-3:
                print("FAIL %s/%s: slice1 check c=%.5f drift=%.2e (per-slice "
                      "normalization suspected) -- NOT topping up" % (tax, rec, c1, drift))
                failed += 1
                continue

        max_sl = os.path.getsize(spath) // (ss * BYTES_PER_SAMP[kind])
        tgt = min(TARGET, int(max_sl))
        added = 0
        for si in range(n_cur, tgt):
            x = alpha0 * read_seg(spath, kind, si * ss, ss)
            if len(x) < ss:
                break
            out = np.empty(2 * ss, dtype=np.float32)
            out[0::2] = x.real.astype(np.float32)
            out[1::2] = x.imag.astype(np.float32)
            out.tofile(os.path.join(rdir, "slice_%04d.iq" % si))
            added += 1
        md["n_slices"] = n_cur + added
        md["n_slices_available"] = int(max_sl)
        md["topup_source"] = os.path.relpath(spath, RFA_ROOT)
        md["topup_reader"] = kind
        md["topup_scale"] = [alpha0.real, alpha0.imag]
        md["topup_corr"] = round(c0, 6)
        json.dump(md, open(mpath, "w", encoding="utf-8"), indent=2)
        new_slices += added
        done += 1
        print("OK %-22s %-40s %s c=%.5f alpha=%.3e +%d -> %d"
              % (tax[:22], rec[:40], kind, c0, abs(alpha0), added, n_cur + added))

print("topup done: recs=%d skipped=%d failed=%d new_slices=%d"
      % (done, skipped, failed, new_slices))
if failed:
    raise SystemExit(1)
