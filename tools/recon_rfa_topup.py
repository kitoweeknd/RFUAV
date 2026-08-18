"""Recon for DroneRFa top-up: map every canonical DroneRFa recording to its
source file on disk and compute how many 0.1s slices the source can yield."""
import os, json, glob

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
RFA_ROOT = r"E:\Drone_dataset\RFA"

# index all candidate source files under RFA (stem -> [paths])
index = {}
for dirpath, dirs, files in os.walk(RFA_ROOT):
    for fn in files:
        if fn.lower().endswith((".dat", ".bin", ".iq", ".mat")):
            stem = os.path.splitext(fn)[0]
            index.setdefault(stem, []).append(os.path.join(dirpath, fn))

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
        slices = glob.glob(os.path.join(tdir, rec, "slice_*.iq"))
        stem = md.get("original_file", "?")
        cands = index.get(stem, [])
        ss = md.get("slice_samples", 0)
        maxsl = "?"
        src_sz = 0
        if cands:
            src_sz = os.path.getsize(cands[0])
            cplx = stem.endswith("_complex")
            bytes_per_samp = 8  # complex64 on disk OR 2x float32 interleaved
            maxsl = src_sz // (ss * bytes_per_samp) if ss else "?"
        recs.append((tax, rec, stem, len(slices), ss, md.get("sample_rate_hz"),
                     len(cands), maxsl, round(src_sz / 1e6)))

print("DroneRFa recordings in canonical: %d" % len(recs))
print("%-24s %-38s cur ss(M)   fs(M) src max srcMB" % ("tax", "rec"))
n_missing = n_short = 0
for tax, rec, stem, cur, ss, fs, nc, maxsl, mb in recs:
    flag = ""
    if nc == 0:
        flag = " <-- NO SOURCE"; n_missing += 1
    elif isinstance(maxsl, int) and maxsl < 10:
        flag = " <-- short (max %s)" % maxsl; n_short += 1
    print("%-24s %-38s %3d %5.1f %7.1f %3d %4s %5d%s"
          % (tax[:24], rec[:38], cur, ss / 1e6 if ss else 0,
             (fs or 0) / 1e6, nc, maxsl, mb, flag))
print("missing_source=%d short=%d" % (n_missing, n_short))
# show duplicate-candidate stems
for tax, rec, stem, cur, ss, fs, nc, maxsl, mb in recs:
    if nc > 1:
        print("DUP %s -> %s" % (stem, index[stem]))
