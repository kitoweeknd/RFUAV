"""Import DroneRFb-DIR .mat (v7.3) files into canonical tree.

Structure: train/{code}_{IN|OUT}_{scene}_slice_{n}.mat with I/Q float32 arrays.
Codes: A1,A2,C1,C2,D1,D2,E1,E2,F1,F2,G1,G2 (6 models x 2 individuals) + background.

Grouping: each (code, IN/OUT, scene) combination = one recording session.
Slices within a session stay together (leakage-safe grouping preserved).
"""
import os, sys, json, time, re
import numpy as np
import h5py

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

SRC = r"E:\Drone_dataset\DroneRFb\twin_droneRF\train"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
LOG = r"D:\ML\RFUAV\tools\import_dronerfb_log.txt"
FS = 100e6  # DroneRFb-DIR: 100 MSps USRP capture

# 4M samples per .mat slice = 0.04 s at 100 MHz
MAX_SLICES_PER_SESSION = 3   # cap: 3 slices (.mat files) per session
                             # keeps disk usage bounded; sessions have 20-60 slices

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "%s  %s" % (ts, msg)
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=== DroneRFb-DIR import start ===")

# Group files by session: {code}_{IN/OUT}[_{scene}]
files = [f for f in os.listdir(SRC) if f.endswith(".mat")]
log("Total .mat files: %d" % len(files))

sessions = {}
pat = re.compile(r"^([A-G][12]|background)_(IN|OUT)(?:_(S\d+))?_slice_(\d+)\.mat$")
unparsed = 0
for f in files:
    m = pat.match(f)
    if not m:
        unparsed += 1
        continue
    code, inout, scene, slice_n = m.group(1), m.group(2), m.group(3) or "S0", int(m.group(4))
    key = (code, inout, scene)
    sessions.setdefault(key, []).append((slice_n, f))

log("Sessions: %d  (unparsed files: %d)" % (len(sessions), unparsed))

imported = 0
skipped = 0
for (code, inout, scene), slist in sorted(sessions.items()):
    tax_id = "ext.dronerfb." + code.lower()
    rec_stem = "drfb_%s_%s_%s" % (code, inout, scene)
    rec_dir = os.path.join(CANON, tax_id, rec_stem)

    if os.path.exists(os.path.join(rec_dir, "metadata.json")):
        skipped += 1
        continue

    os.makedirs(rec_dir, exist_ok=True)
    slist.sort()  # by slice number

    native_snr = None
    written = 0
    for slice_n, fname in slist[:MAX_SLICES_PER_SESSION]:
        fpath = os.path.join(SRC, fname)
        try:
            with h5py.File(fpath, "r") as hf:
                I = np.array(hf["I"]).ravel().astype(np.float32)
                Q = np.array(hf["Q"]).ravel().astype(np.float32)
        except Exception as e:
            log("  READ FAIL %s: %s" % (fname, e))
            continue

        n = min(len(I), len(Q))
        iq = np.empty(n, dtype=np.complex64)
        iq.real = I[:n]
        iq.imag = Q[:n]

        if np.max(np.abs(iq)) < 1e-15:
            log("  EMPTY %s" % fname)
            continue

        # SNR on first slice only
        if native_snr is None:
            fcme = FCMEEstimator(fs=FS, nfft=min(16384, n))
            snr_db, diag = fcme.estimate(iq)
            native_snr = round(float(snr_db), 1) if not np.isnan(snr_db) else None

        # Write as float32 interleaved .iq (canonical format)
        data = np.empty(2 * n, dtype=np.float32)
        data[0::2] = iq.real
        data[1::2] = iq.imag
        data.tofile(os.path.join(rec_dir, "slice_%04d.iq" % written))
        written += 1

    if written == 0:
        log("  SESSION EMPTY: %s" % rec_stem)
        continue

    metadata = {
        "source_dataset": "DroneRFb-DIR",
        "taxonomy_id": tax_id,
        "individual_code": code,
        "environment": inout,
        "scene": scene,
        "native_snr_db": native_snr,
        "sample_rate_hz": FS,
        "n_slices": written,
        "n_slices_available": len(slist),
        "original_format": "mat_v7.3_IQ_split",
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    imported += 1
    log("[%d] %-28s %s  SNR=%s dB  slices=%d/%d" % (
        imported, tax_id, rec_stem, native_snr, written, len(slist)))

# Summary
total_classes = len([c for c in os.listdir(CANON) if os.path.isdir(os.path.join(CANON, c))])
total_recs = sum(len([r for r in os.listdir(os.path.join(CANON, c))
                      if os.path.isdir(os.path.join(CANON, c, r))])
                 for c in os.listdir(CANON) if os.path.isdir(os.path.join(CANON, c)))
log("=== DONE: imported %d sessions (skipped %d existing) ===" % (imported, skipped))
log("=== CANONICAL TOTAL: %d classes, %d recordings ===" % (total_classes, total_recs))
