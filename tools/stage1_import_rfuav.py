"""Stage 1: materialize RFUAV's own rawdata recordings into the canonical tree.

Reads assignment.csv (P0 grouped split), takes every uncut rawdata recording
(train/test/excluded), writes the first CAP 0.1s slices as float32 interleaved
.iq plus a metadata.json sidecar carrying the assignment fields.  Reuses an
existing slice_0000_snr.json when present (same FCME definition), otherwise
estimates SNR on slice_0000.
"""
import os, sys, io, csv, json, time
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

ASSIGN = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
RAW_BASE = r"E:\Drone_dataset\RFUAV"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
FS = 100e6
SLICE_SAMPLES = int(FS * 0.1)   # 10M complex
CAP = 3
FCME_MAX = 5_000_000

def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

rows = []
with io.open(ASSIGN, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("source_root") != "rawdata":
            continue
        if r.get("is_uncut") not in ("1", "True", "true"):
            continue
        if r.get("split") not in ("train", "test", "excluded"):
            continue
        rows.append(r)

log("Stage 1 start: %d uncut rawdata recordings in assignment" % len(rows))

ok = skip = miss = err = 0
for i, r in enumerate(rows, 1):
    tax = r["taxonomy_id"]
    rel = r["source_relpath"]
    stem = os.path.splitext(os.path.basename(rel.replace("/", "\\")))[0]
    src = os.path.normpath(os.path.join(RAW_BASE, rel))
    rec_dir = os.path.join(CANON, tax, stem)
    meta_path = os.path.join(rec_dir, "metadata.json")

    if not os.path.exists(src):
        miss += 1
        log("[%d/%d] MISSING %s" % (i, len(rows), rel))
        continue

    total = os.path.getsize(src) // 8  # complex samples
    avail = int(total // SLICE_SAMPLES)
    want = min(CAP, avail) if avail > 0 else 0
    if want == 0:
        err += 1
        log("[%d/%d] TOO SHORT %s" % (i, len(rows), stem))
        continue

    last_slice = os.path.join(rec_dir, "slice_%04d.iq" % (want - 1))
    if os.path.exists(meta_path) and os.path.exists(last_slice):
        skip += 1
        continue

    os.makedirs(rec_dir, exist_ok=True)
    try:
        with open(src, "rb") as fh:
            for si in range(want):
                out = os.path.join(rec_dir, "slice_%04d.iq" % si)
                data = fh.read(SLICE_SAMPLES * 8)
                if len(data) < SLICE_SAMPLES * 8:
                    break
                if not os.path.exists(out):
                    with open(out, "wb") as dst:
                        dst.write(data)
    except Exception as e:
        err += 1
        log("[%d/%d] READ FAIL %s: %s" % (i, len(rows), stem, e))
        continue

    # SNR: reuse sidecar if present
    snr_path = os.path.join(rec_dir, "slice_0000_snr.json")
    snr_info = None
    if os.path.exists(snr_path):
        try:
            with open(snr_path) as f:
                snr_info = json.load(f)
        except Exception:
            snr_info = None
    if snr_info is None or "native_snr_db" not in snr_info:
        raw = np.fromfile(os.path.join(rec_dir, "slice_0000.iq"),
                          dtype=np.float32, count=FCME_MAX * 2)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw
        fcme = FCMEEstimator(fs=FS, nfft=16384)
        snr_db, diag = fcme.estimate(iq)
        snr_info = {"native_snr_db": round(float(snr_db), 2),
                    "signal_bins": diag["signal_bins"],
                    "noise_bins": diag["noise_bins"],
                    "iterations": diag["iterations"],
                    "T_factor": diag.get("T_factor", 0)}
        with open(snr_path, "w") as f:
            json.dump(snr_info, f, indent=2)

    n_written = sum(1 for f in os.listdir(rec_dir)
                    if f.startswith("slice_") and f.endswith(".iq")
                    and len(f) == len("slice_0000.iq"))
    metadata = {
        "source_dataset": "RFUAV",
        "taxonomy_id": tax,
        "recording_id": r.get("recording_id", ""),
        "session_id": r.get("session_id", ""),
        "capture_condition": r.get("capture_condition", ""),
        "split": r.get("split", ""),
        "fold": r.get("fold", ""),
        "eval_tier": r.get("eval_tier", ""),
        "native_snr_db": snr_info.get("native_snr_db"),
        "sample_rate_hz": FS,
        "n_slices": n_written,
        "n_slices_available": avail,
        "source_relpath": rel,
        "source_size_bytes": int(r.get("size_bytes") or 0),
        "delivered_duration_s": float(r.get("delivered_duration_s") or 0),
        "original_format": "iq_float32_interleaved",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    ok += 1
    if ok % 20 == 0 or i == len(rows):
        log("[%d/%d] progress: ok=%d skip=%d miss=%d err=%d (latest %s SNR=%s)" % (
            i, len(rows), ok, skip, miss, err, stem[:40], snr_info.get("native_snr_db")))

log("Stage 1 DONE: ok=%d skip=%d missing=%d err=%d" % (ok, skip, miss, err))
if miss + err > len(rows) * 0.2:
    sys.exit(1)
