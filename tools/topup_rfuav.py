"""Top up every canonical RFUAV recording toward the TARGET slice count.

Per recording: verify slice_0000/0001 reproduce from the source .iq at
contiguous-from-zero offsets (corr > 0.9999, alpha ~= 1), then append the
missing slices as exact byte-range copies of the source (float32 interleaved,
no numeric round trip). Originals READ-ONLY; writes only under canonical.
"""
import os, json, glob
import numpy as np

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
ROOTS = [r"E:\Drone_dataset\RFUAV", r"E:\Drone_dataset"]
TARGET = 20
NCHK = 200_000
BUF = 8 * 1024 * 1024


def read_f32(path, start, n):
    raw = np.fromfile(path, dtype=np.float32, count=2 * n, offset=start * 8)
    raw = raw.astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) / d) if d > 0 else 0.0


def copy_range(src, dst, byte_off, nbytes):
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fi.seek(byte_off)
        left = nbytes
        while left > 0:
            chunk = fi.read(min(BUF, left))
            if not chunk:
                break
            fo.write(chunk)
            left -= len(chunk)
    return nbytes - left


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
        if md.get("source_dataset", "RFUAV") != "RFUAV":
            continue
        cur = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))
        n_cur = len(cur)
        if n_cur >= TARGET:
            skipped += 1
            continue
        if not cur:
            print("FAIL %s/%s: no existing slices" % (tax, rec))
            failed += 1
            continue
        ss = os.path.getsize(cur[0]) // 8

        src = None
        rel = md.get("source_relpath", "")
        for root in ROOTS:
            p = os.path.normpath(os.path.join(root, rel))
            if rel and os.path.isfile(p):
                src = p
                break
        if src is None:
            print("FAIL %s/%s: source not found (%r)" % (tax, rec, rel[:60]))
            failed += 1
            continue

        a0 = read_f32(cur[0], 0, NCHK)
        b0 = read_f32(src, 0, NCHK)
        c0 = corr(a0, b0)
        alpha = complex(np.vdot(b0, a0) / np.vdot(b0, b0)) if c0 > 0.99 else 0
        c1 = 1.0
        if n_cur > 1:
            c1 = corr(read_f32(cur[1], 0, NCHK), read_f32(src, ss, NCHK))
        if c0 < 0.9999 or c1 < 0.9999 or abs(alpha - 1) > 1e-6:
            print("FAIL %s/%s: verify c0=%.5f c1=%.5f alpha=%s" % (tax, rec, c0, c1, alpha))
            failed += 1
            continue

        src_sz = os.path.getsize(src)
        max_sl = src_sz // (ss * 8)
        tgt = min(TARGET, int(max_sl))
        added = 0
        for si in range(n_cur, tgt):
            out = os.path.join(rdir, "slice_%04d.iq" % si)
            got = copy_range(src, out, si * ss * 8, ss * 8)
            if got != ss * 8:
                os.remove(out)
                break
            added += 1
        md["n_slices"] = n_cur + added
        md["n_slices_available"] = int(max_sl)
        md["slice_samples"] = int(ss)
        md["topup_reader"] = "f32_bytecopy"
        md["topup_corr"] = round(c0, 6)
        json.dump(md, open(mpath, "w", encoding="utf-8"), indent=2)
        new_slices += added
        done += 1
        print("OK %-24s %-46s +%d -> %d (avail %d)"
              % (tax[:24], rec[:46], added, n_cur + added, max_sl))

print("rfuav topup done: recs=%d skipped=%d failed=%d new_slices=%d (~%.1f GB)"
      % (done, skipped, failed, new_slices, new_slices * 0.08))
if failed:
    raise SystemExit(1)
