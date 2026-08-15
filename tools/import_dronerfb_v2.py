"""Import remaining DroneRFb-DIR data into canonical tree (round 2).

Covers what round 1 (import_dronerfb.py) skipped:
1. train/background_slice_{n}.mat  -> ext.dronerfb.background (no IN/OUT token)
2. test/{n}.mat via test_labels.txt mapping "ORIGNAME n" -> recovers original
   names incl. THIRD individuals (A3/C3/D3/E3/F3/G3) absent from train/.
   X1/X2 sessions already imported from train/ are skipped (same recordings,
   different slices - slice-level overlap must not create duplicate sessions).

Session key = (code, IN/OUT, scene); one canonical recording per session.
"""
import os, sys, json, time, re
import numpy as np
import h5py

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

BASE = r"E:\Drone_dataset\DroneRFb\twin_droneRF"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
LOG = r"D:\ML\RFUAV\tools\import_dronerfb_v2_log.txt"
FS = 100e6
MAX_SLICES_PER_SESSION = 3

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "%s  %s" % (ts, msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def read_iq(fpath):
    with h5py.File(fpath, "r") as hf:
        I = np.array(hf["I"]).ravel().astype(np.float32)
        Q = np.array(hf["Q"]).ravel().astype(np.float32)
    n = min(len(I), len(Q))
    iq = np.empty(n, dtype=np.complex64)
    iq.real = I[:n]
    iq.imag = Q[:n]
    return iq

def import_session(key, slist, origin_split):
    """slist: list of (slice_n, disk_path, original_name)."""
    code, inout, scene = key
    tax_id = "ext.dronerfb." + code.lower()
    rec_stem = "drfb_%s" % code if inout == "NA" else "drfb_%s_%s_%s" % (code, inout, scene)
    rec_dir = os.path.join(CANON, tax_id, rec_stem)

    if os.path.exists(os.path.join(rec_dir, "metadata.json")):
        return "skip", tax_id, rec_stem, None, 0, len(slist)

    os.makedirs(rec_dir, exist_ok=True)
    slist.sort()
    native_snr = None
    written = 0
    src_files = []
    for slice_n, dpath, oname in slist[:MAX_SLICES_PER_SESSION]:
        try:
            iq = read_iq(dpath)
        except Exception as e:
            log("  READ FAIL %s: %s" % (dpath, e))
            continue
        if iq.size == 0 or np.max(np.abs(iq)) < 1e-15:
            log("  EMPTY %s" % dpath)
            continue
        if native_snr is None:
            fcme = FCMEEstimator(fs=FS, nfft=min(16384, iq.size))
            snr_db, diag = fcme.estimate(iq)
            native_snr = round(float(snr_db), 1) if not np.isnan(snr_db) else None
        data = np.empty(2 * iq.size, dtype=np.float32)
        data[0::2] = iq.real
        data[1::2] = iq.imag
        data.tofile(os.path.join(rec_dir, "slice_%04d.iq" % written))
        src_files.append([os.path.basename(dpath), oname])
        written += 1

    if written == 0:
        log("  SESSION EMPTY: %s" % rec_stem)
        return "empty", tax_id, rec_stem, None, 0, len(slist)

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
        "origin_split": origin_split,
        "source_files": src_files,
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return "ok", tax_id, rec_stem, native_snr, written, len(slist)

log("=== DroneRFb-DIR import round 2 start ===")

# --- Part 1: background from train/ ---
train_dir = os.path.join(BASE, "train")
bg_pat = re.compile(r"^background_slice_(\d+)\.mat$")
bg = []
for f in os.listdir(train_dir):
    m = bg_pat.match(f)
    if m:
        bg.append((int(m.group(1)), os.path.join(train_dir, f), f))
log("Part 1: background files in train/: %d" % len(bg))
if bg:
    st, tax, stem, snr, w, avail = import_session(("background", "NA", "S0"), bg, "train")
    log("[bg] %s %s %s SNR=%s slices=%d/%d" % (st, tax, stem, snr, w, avail))

# --- Part 2: test/ via label mapping ---
test_dir = os.path.join(BASE, "test")
label_path = os.path.join(BASE, "test_labels.txt")
pat = re.compile(r"^([A-G]\d|background)(?:_(IN|OUT))?(?:_(S\d+))?_slice_(\d+)\.mat$")

sessions = {}
n_lines = n_missing = n_unparsed = 0
with open(label_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        n_lines += 1
        parts = line.split()
        if len(parts) != 2:
            n_unparsed += 1
            continue
        oname, num = parts
        m = pat.match(oname)
        if not m:
            n_unparsed += 1
            log("  UNPARSED LABEL: %s" % line)
            continue
        code = m.group(1)
        inout = m.group(2) or "NA"
        scene = m.group(3) or "S0"
        slice_n = int(m.group(4))
        dpath = os.path.join(test_dir, num + ".mat")
        if not os.path.exists(dpath):
            n_missing += 1
            continue
        sessions.setdefault((code, inout, scene), []).append((slice_n, dpath, oname))

log("Part 2: label lines=%d  sessions=%d  missing_disk=%d  unparsed=%d" % (
    n_lines, len(sessions), n_missing, n_unparsed))
codes = sorted(set(k[0] for k in sessions))
log("  codes in test/: %s" % ", ".join(codes))

imported = skipped = 0
for key in sorted(sessions):
    st, tax, stem, snr, w, avail = import_session(key, sessions[key], "test")
    if st == "ok":
        imported += 1
        log("[%d] %-26s %s  SNR=%s dB  slices=%d/%d" % (imported, tax, stem, snr, w, avail))
    elif st == "skip":
        skipped += 1

log("=== round 2: imported %d new sessions, skipped %d existing ===" % (imported, skipped))

# Summary
classes = [c for c in os.listdir(CANON) if os.path.isdir(os.path.join(CANON, c))]
total_recs = sum(len([r for r in os.listdir(os.path.join(CANON, c))
                      if os.path.isdir(os.path.join(CANON, c, r))]) for c in classes)
log("=== CANONICAL TOTAL: %d classes, %d recordings ===" % (len(classes), total_recs))
