"""Import the 3 DroneRF recordings missed by round 1 (multi-part rars with
inconsistent upstream names, incl. an 'FR Data_' typo):
  Phantom 11000_L  = RF Data_11000_L1.rar + L2
  Background 00000_H = RF Data_00000_H1.rar + FR Data_00000_H2.rar
  Background 00000_L = RF Data_00000_L1.rar + L2
Parts are extracted together and CSV segments merged by segment index
(duplicate indices across parts -> hard fail for manual review).
Slices are produced directly in the final 20 MHz decimated form.
"""
import os, re, sys, json, glob, time, shutil, subprocess
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from scipy.signal import hilbert
from snr_dual import FCMEEstimator

RF_ROOT = r"E:\Drone_dataset\RF"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
STAGING = r"E:\Drone_dataset\RFUAV-v2\staging\dronerf"
SEVENZ = r"C:\Program Files\7-Zip\7z.exe"
FS_RAW, SS_RAW = 40e6, 4_000_000
FS_OUT, SS_OUT = 20e6, 2_000_000
TARGET = 20

JOBS = [
    ("Phantom drone", "uav.dji.phantom3", "11000", "L",
     ["RF Data_11000_L1.rar", "RF Data_11000_L2.rar"]),
    ("Background RF activites", "bg.dronerf", "00000", "H",
     ["RF Data_00000_H1.rar", "FR Data_00000_H2.rar"]),
    ("Background RF activites", "bg.dronerf", "00000", "L",
     ["RF Data_00000_L1.rar", "RF Data_00000_L2.rar"]),
]


def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


shift = np.exp(-1j * 2 * np.pi * 10e6 * np.arange(SS_RAW) / FS_RAW).astype(np.complex64)

done = failed = 0
for cdir, tax, bui, band, parts in JOBS:
    rec = "dronerf_%s_%s" % (bui, band)
    rdir = os.path.join(CANON, tax, rec)
    if os.path.exists(os.path.join(rdir, "metadata.json")):
        log("SKIP %s (exists)" % rec)
        continue
    if os.path.isdir(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING)
    ok = True
    for p in parts:
        rc = subprocess.call([SEVENZ, "x", "-y", "-o" + STAGING,
                              os.path.join(RF_ROOT, cdir, p)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            log("EXTRACT FAIL %s rc=%d" % (p, rc))
            ok = False
    if not ok:
        failed += 1
        continue
    csvs = glob.glob(os.path.join(STAGING, "**", "*.csv"), recursive=True)
    by_idx = {}
    dup = False
    for c in csvs:
        m = re.search(r"_(\d+)\.csv$", os.path.basename(c))
        idx = int(m.group(1)) if m else -1
        if idx in by_idx:
            log("DUPLICATE segment idx %d: %s vs %s" % (idx, by_idx[idx], c))
            dup = True
        by_idx[idx] = c
    if dup:
        log("FAIL %s: overlapping segment indices across parts" % rec)
        failed += 1
        continue
    log("%s: %d csv segments (idx %d..%d)" % (rec, len(by_idx),
                                              min(by_idx), max(by_idx)))
    os.makedirs(rdir, exist_ok=True)
    buf = np.empty(0, dtype=np.float32)
    n_slices = 0
    total_real = 0
    bps = None
    for idx in sorted(by_idx):
        cpath = by_idx[idx]
        if n_slices >= TARGET and bps is not None:
            total_real += int(os.path.getsize(cpath) / bps)
            continue
        x = np.fromfile(cpath, sep=",", dtype=np.float64).astype(np.float32)
        if bps is None and len(x):
            bps = os.path.getsize(cpath) / len(x)
        total_real += len(x)
        buf = np.concatenate([buf, x]) if len(buf) else x
        while len(buf) >= SS_RAW and n_slices < TARGET:
            seg = buf[:SS_RAW]
            buf = buf[SS_RAW:]
            y = (hilbert(seg.astype(np.float64)).astype(np.complex64) * shift)
            y2 = np.ascontiguousarray(y[::2]).astype(np.complex64)
            out = np.empty(2 * SS_OUT, dtype=np.float32)
            out[0::2] = y2.real
            out[1::2] = y2.imag
            out.tofile(os.path.join(rdir, "slice_%04d.iq" % n_slices))
            n_slices += 1
    if n_slices == 0:
        log("FAIL %s: no slices" % rec)
        shutil.rmtree(rdir, ignore_errors=True)
        failed += 1
        continue
    raw = np.fromfile(os.path.join(rdir, "slice_0000.iq"), dtype=np.float32,
                      count=2 * SS_OUT)
    iq = np.empty(SS_OUT, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    snr, diag = FCMEEstimator(fs=FS_OUT, nfft=16384).estimate(iq)
    meta = {
        "source_dataset": "DroneRF",
        "taxonomy_id": tax,
        "session_id": "dronerf_bui_%s" % bui,
        "mode_bui": bui,
        "band": band,
        "native_snr_db": round(float(snr), 1) if np.isfinite(snr) else None,
        "sample_rate_hz": FS_OUT,
        "slice_samples": SS_OUT,
        "n_slices": n_slices,
        "n_slices_available": int(total_real // SS_RAW),
        "original_format": "csv_text_real_40MSps (multi-part rar)",
        "real_to_complex": "hilbert analytic + (-10 MHz) shift + 2:1 "
                           "decimation (lossless, half-band was empty), "
                           "complex64 @ 20 MHz",
        "snr_method": "fcme_after_decimation",
        "source_relpath": "RF\\%s\\{%s}" % (cdir, ",".join(parts)),
    }
    json.dump(meta, open(os.path.join(rdir, "metadata.json"), "w",
                         encoding="utf-8"), indent=2)
    done += 1
    log("OK %-18s %-18s slices=%d/%d SNR=%s dB" % (
        tax[:18], rec, n_slices, meta["n_slices_available"], meta["native_snr_db"]))

if os.path.isdir(STAGING):
    shutil.rmtree(STAGING)
log("round-2 import done: recs=%d failed=%d" % (done, failed))
if failed:
    sys.exit(1)
