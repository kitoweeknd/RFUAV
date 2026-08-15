"""Merge split-volume DroneRFb zip, extract, and import into canonical."""
import os, sys, json, time, zipfile, numpy as np, shutil, tempfile

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from graphic.RawDataProcessor import generate_images

DST = r"E:\Drone_dataset\DroneRFb"
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
LOG = r"D:\ML\RFUAV\tools\pipeline_dronerfb_log.txt"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = "%s  %s" % (ts, msg)
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("=== Merge DroneRFb split volumes ===")

# ── Step 1: Merge split volumes into one zip ──
merged_zip = os.path.join(DST, "merged_full.zip")
if not os.path.exists(merged_zip):
    log("Merging split volumes...")
    # Collect parts: .z01 to .z31, then .zip
    parts = []
    for i in range(1, 32):  # z01 to z31
        p = os.path.join(DST, "twin_droneRF.z%02d" % i)
        if os.path.exists(p):
            parts.append(p)
    parts.append(os.path.join(DST, "twin_droneRF.zip"))  # main zip file

    log("Parts: %d" % len(parts))
    total_sz = sum(os.path.getsize(p) for p in parts)
    log("Total size: %.1f GB" % (total_sz / 1e9))

    with open(merged_zip, "wb") as out:
        for p in parts:
            sz = os.path.getsize(p)
            with open(p, "rb") as f:
                copied = 0
                while True:
                    chunk = f.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    copied += len(chunk)
            log("  Merged %s: %.1f MB" % (os.path.basename(p), sz / 1e6))
    log("Merged OK: %.1f GB" % (os.path.getsize(merged_zip) / 1e9))

# ── Step 2: Extract ──
extracted = os.path.join(DST, "extracted")
if not os.path.exists(extracted):
    os.makedirs(extracted, exist_ok=True)
    log("Extracting...")
    try:
        with zipfile.ZipFile(merged_zip, "r") as zf:
            zf.extractall(extracted)
        log("Extracted OK")
    except Exception as e:
        log("Extract error: %s" % e)
        raise

# ── Step 3: Probe ──
log("Probing extracted files...")
all_files = []
for root, dirs, files in os.walk(extracted):
    for f in files:
        all_files.append(os.path.join(root, f))

log("Total files: %d" % len(all_files))

# Identify classes from file naming
from collections import Counter
cls_counter = Counter()
for f in all_files:
    rel = os.path.relpath(f, extracted)
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:
        cls_counter[parts[0]] += 1

log("Classes found: %d" % len(cls_counter))
for c, n in cls_counter.most_common(20):
    log("  %s: %d files" % (c, n))

# Class name mapping
CLASS_MAP = {
    "avata": "uav.dji.avata2", "avata2": "uav.dji.avata2",
    "air2s": "uav.dji.air2s", "air2": "uav.dji.air2s",
    "mavic3": "uav.dji.mavic3", "mavic": "uav.dji.mavic3",
    "mini3": "uav.dji.mini3", "mini3pro": "uav.dji.mini3_pro",
    "phantom": "uav.dji.phantom4",
    "matrice": "uav.dji.matrice300",
    "inspire": "uav.dji.inspire2",
    "frsky": "rc.frsky.x20",
    "futaba": "rc.futaba.t14sg",
    "radiolink": "rc.radiolink.at9s",
    "yunzhuo": "rc.yunzhuo.h16",
    "jumper": "rc.jumper.t14",
    "radiomaster": "rc.radiomaster.boxer",
    "flysky": "rc.flysky.el18",
    "siyi": "rc.siyi.mk32",
    "skydroid": "rc.skydroid.t10",
    "wfly": "rc.wfly.et10",
}

def map_class(name):
    n = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for key, tax_id in CLASS_MAP.items():
        if key in n:
            return tax_id
    return "ext." + name.lower().replace(" ", "_")[:25]

# ── Step 4: Import ──
log("Importing into canonical...")
imported = 0
for fpath in sorted(all_files):
    if not fpath.lower().endswith((".dat", ".iq", ".bin")):
        continue
    rel = os.path.relpath(fpath, extracted)
    parts = rel.replace("\\", "/").split("/")
    cls_name = parts[0] if len(parts) >= 2 else "unknown"
    tax_id = map_class(cls_name)
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    rec_stem = "drfb_" + stem[:40]
    rec_dir = os.path.join(CANON, tax_id, rec_stem)

    if os.path.exists(os.path.join(rec_dir, "metadata.json")):
        continue

    sz_mb = os.path.getsize(fpath) / 1e6
    if sz_mb < 0.01:  # skip tiny files
        continue

    os.makedirs(rec_dir, exist_ok=True)

    # Try float32 first
    try:
        raw = np.fromfile(fpath, dtype=np.float32, count=min(5_000_000 * 2, int(sz_mb * 1e6 // 4)))
        if raw.max() > 1e-15 and raw.size >= 2:
            n = raw.size // 2
            iq = np.empty(n, dtype=np.complex64)
            iq.real = raw[0::2]; iq.imag = raw[1::2]
            fs = 20e6
        else:
            raise ValueError("no signal")
    except:
        # Try complex64
        raw = np.fromfile(fpath, dtype=np.complex64, count=min(5_000_000, int(sz_mb * 1e6 // 8)))
        if np.max(np.abs(raw)) > 1e-15:
            iq = raw
            fs = 20e6
        else:
            log("  SKIP %s: no signal detected" % fname)
            continue

    fcme = FCMEEstimator(fs=fs, nfft=min(16384, len(iq)))
    snr_db, diag = fcme.estimate(iq[:min(len(iq), 5_000_000)])

    slice_s = int(fs * 0.1)
    n_slices = len(iq) // slice_s
    n_slices = min(n_slices, 2)

    metadata = {
        "source_dataset": "DroneRFb",
        "taxonomy_id": tax_id,
        "native_snr_db": round(snr_db, 1) if not np.isnan(snr_db) else None,
        "sample_rate_hz": fs,
        "n_slices": n_slices,
        "original_file": fname,
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    for si in range(n_slices):
        start = si * slice_s; end = start + slice_s
        if end > len(iq): break
        sl = iq[start:end]
        data = np.empty(2 * len(sl), dtype=np.float32)
        data[0::2] = sl.real; data[1::2] = sl.imag
        data.tofile(os.path.join(rec_dir, "slice_%04d.iq" % si))

    imported += 1
    log("  [%d] %-30s SNR=%.1f dB  slices=%d  fs=%.0f MHz" % (imported, tax_id, snr_db, n_slices, fs/1e6))

# ── Step 5: Summary ──
total_classes = len([c for c in os.listdir(CANON) if os.path.isdir(os.path.join(CANON, c))])
total_recs = sum(len([r for r in os.listdir(os.path.join(CANON, c)) if os.path.isdir(os.path.join(CANON, c, r))]) for c in os.listdir(CANON))
log("=== DONE: %d classes, %d recordings ===" % (total_classes, total_recs))
