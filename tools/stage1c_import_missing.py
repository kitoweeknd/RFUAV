"""Stage 1c: import the 18 uncut recordings stage1 missed (non-'rawdata'
source_root, e.g. the second-batch Chinese-named root), then quarantine the
30 sidecar-only orphan dirs superseded by these imports.
"""
import os, sys, io, csv, json, time, shutil
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

ASSIGN = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
RAW_BASE = r"E:\Drone_dataset\RFUAV"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
QUAR = r"E:\Drone_dataset\RFUAV-v2\quarantine\canonical_orphans"
FS = 100e6
SLICE_SAMPLES = int(FS * 0.1)
CAP = 3
FCME_MAX = 5_000_000

def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

rows = []
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("is_uncut") != "1":
            continue
        if r.get("split") not in ("train", "test", "excluded"):
            continue
        stem = os.path.splitext(os.path.basename(r["source_relpath"].replace("/", "\\")))[0]
        if os.path.exists(os.path.join(CANON, r["taxonomy_id"], stem, "metadata.json")):
            continue
        rows.append((r, stem))

log("Stage 1c: %d missing uncut recordings" % len(rows))
ok = err = 0
for r, stem in rows:
    tax = r["taxonomy_id"]
    src = os.path.normpath(os.path.join(RAW_BASE, r["source_relpath"]))
    rec_dir = os.path.join(CANON, tax, stem)
    if not os.path.exists(src):
        err += 1
        log("MISSING SRC %s" % src)
        continue
    total = os.path.getsize(src) // 8
    avail = int(total // SLICE_SAMPLES)
    want = min(CAP, avail)
    if want == 0:
        err += 1
        log("TOO SHORT %s" % stem)
        continue
    os.makedirs(rec_dir, exist_ok=True)
    with open(src, "rb") as fh:
        for si in range(want):
            data = fh.read(SLICE_SAMPLES * 8)
            if len(data) < SLICE_SAMPLES * 8:
                break
            out = os.path.join(rec_dir, "slice_%04d.iq" % si)
            if not os.path.exists(out):
                with open(out, "wb") as dst:
                    dst.write(data)
    raw = np.fromfile(os.path.join(rec_dir, "slice_0000.iq"),
                      dtype=np.float32, count=FCME_MAX * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw
    fcme = FCMEEstimator(fs=FS, nfft=16384)
    snr_db, diag = fcme.estimate(iq)
    snr_info = {"native_snr_db": round(float(snr_db), 2),
                "signal_bins": diag["signal_bins"], "noise_bins": diag["noise_bins"],
                "iterations": diag["iterations"], "T_factor": diag.get("T_factor", 0)}
    with open(os.path.join(rec_dir, "slice_0000_snr.json"), "w") as f:
        json.dump(snr_info, f, indent=2)
    n_written = sum(1 for f in os.listdir(rec_dir)
                    if f.startswith("slice_") and f.endswith(".iq") and len(f) == 13)
    metadata = {
        "source_dataset": "RFUAV", "taxonomy_id": tax,
        "recording_id": r.get("recording_id", ""), "session_id": r.get("session_id", ""),
        "capture_condition": r.get("capture_condition", ""),
        "split": r.get("split", ""), "fold": r.get("fold", ""),
        "eval_tier": r.get("eval_tier", ""),
        "native_snr_db": snr_info["native_snr_db"], "sample_rate_hz": FS,
        "n_slices": n_written, "n_slices_available": avail,
        "source_relpath": r["source_relpath"],
        "source_root": r.get("source_root", ""),
        "source_size_bytes": int(r.get("size_bytes") or 0),
        "delivered_duration_s": float(r.get("delivered_duration_s") or 0),
        "original_format": "iq_float32_interleaved",
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    ok += 1
    log("[%d/%d] %s / %s  SNR=%.1f  split=%s" % (ok + err, len(rows), tax, stem[:45],
        snr_info["native_snr_db"], r.get("split")))

log("import done: ok=%d err=%d" % (ok, err))

# Quarantine sidecar-only orphan dirs (no metadata.json, no .iq slices)
moved = 0
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        if not os.path.isdir(rdir):
            continue
        if os.path.exists(os.path.join(rdir, "metadata.json")):
            continue
        files = os.listdir(rdir)
        if any(f.endswith(".iq") for f in files):
            log("ORPHAN WITH SLICES (left in place): %s/%s" % (tax, rec))
            continue
        dest = os.path.join(QUAR, tax, rec)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(rdir, dest)
        moved += 1
    # remove now-empty class dirs
    if os.path.isdir(tdir) and not os.listdir(tdir):
        os.rmdir(tdir)
        log("removed empty class dir: %s" % tax)
log("quarantined %d orphan dirs -> %s" % (moved, QUAR))
if err > 3:
    sys.exit(1)
