"""Unattended DroneRFb download + merge pipeline. Runs end-to-end."""
import os, sys, json, time, subprocess, shutil, glob, zipfile, numpy as np

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

log("=== DroneRFb pipeline start ===")

# ── Step 1: Wait for downloads ──
log("Step 1: Checking downloads...")
expected = 32  # 31 z-parts + 1 zip
while True:
    files = [f for f in os.listdir(DST) if os.path.isfile(os.path.join(DST, f))]
    n = len(files)
    log("  Files: %d/%d" % (n, expected))
    if n >= expected:
        break
    time.sleep(300)  # wait 5 min

log("All %d files present" % n)

# ── Step 2: Extract zip ──
log("Step 2: Extracting...")
zip_path = os.path.join(DST, "twin_droneRF.zip")
extracted = os.path.join(DST, "extracted")
if not os.path.exists(zip_path):
    log("ERROR: twin_droneRF.zip not found!")
    raise SystemExit(1)

if not os.path.exists(extracted):
    os.makedirs(extracted, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted)
        log("Extracted %d files" % len(os.listdir(extracted)))
    except Exception as e:
        log("Zip extraction failed: %s (trying 7z)" % e)
        subprocess.run(["7z", "x", zip_path, "-o" + extracted], check=True)

# ── Step 3: Probe format ──
log("Step 3: Probing format...")
extracted_dir = os.path.join(extracted, "twin_droneRFa") if os.path.isdir(os.path.join(extracted, "twin_droneRFa")) else extracted
all_files = []
for root, dirs, files in os.walk(extracted_dir):
    for f in files:
        all_files.append(os.path.join(root, f))

log("Total files: %d" % len(all_files))

# Try to identify drone classes from directory structure
classes = set()
for f in all_files:
    rel = os.path.relpath(f, extracted_dir)
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:
        classes.add(parts[0])

log("Detected classes: %d" % len(classes))
for c in sorted(classes):
    count = len([f for f in all_files if c in f])
    log("  %s: %d files" % (c, count))

# Probe first file
sample = sorted(all_files)[0]
sz = os.path.getsize(sample)
log("Sample file: %s (%.1f MB)" % (os.path.basename(sample), sz / 1e6))

# Try reading as float32 IQ
try:
    raw = np.fromfile(sample, dtype=np.float32, count=100)
    log("  float32 first 10: %s" % raw[:10])
    log("  float32 range: [%.6f, %.6f]" % (raw.min(), raw.max()))
    is_valid = raw.max() > 1e-15
    log("  Valid signal: %s" % is_valid)
except Exception as e:
    log("  float32 read failed: %s" % e)

# Try complex64
try:
    raw = np.fromfile(sample, dtype=np.complex64, count=100)
    log("  complex64 first 5: %s" % raw[:5])
    is_valid_c = np.max(np.abs(raw)) > 1e-15
    log("  complex64 valid: %s" % is_valid_c)
except Exception as e:
    log("  complex64 read failed: %s" % e)

# ── Step 4: Import into canonical ──
log("Step 4: Importing into canonical tree...")

# Map class names to taxonomy_ids
CLASS_MAP = {
    "avata": "uav.dji.avata2",
    "avata2": "uav.dji.avata2",
    "air2s": "uav.dji.air2s",
    "mavic3": "uav.dji.mavic3",
    "mini3": "uav.dji.mini3",
    "mini3pro": "uav.dji.mini3_pro",
    "phantom": "uav.dji.phantom4",
    "matrice": "uav.dji.matrice300",
    "inspire": "uav.dji.inspire2",
    "frsky": "rc.frsky.x20",
    "futaba": "rc.futaba.t14sg",
    "radiolink": "rc.radiolink.at9s",
}

def map_class(dirname):
    dn = dirname.lower().replace("_", "").replace("-", "").replace(" ", "")
    for key, tax_id in CLASS_MAP.items():
        if key in dn:
            return tax_id
    return "ext." + dirname.lower().replace(" ", "_")[:20]

imported = 0
for fpath in sorted(all_files):
    rel = os.path.relpath(fpath, extracted_dir)
    parts = rel.replace("\\", "/").split("/")
    cls_name = parts[0] if len(parts) >= 2 else "unknown"
    tax_id = map_class(cls_name)
    fname = os.path.basename(fpath)
    stem = os.path.splitext(fname)[0]
    rec_stem = "drf_" + stem[:40]
    rec_dir = os.path.join(CANON, tax_id, rec_stem)

    if os.path.exists(os.path.join(rec_dir, "metadata.json")):
        continue

    os.makedirs(rec_dir, exist_ok=True)
    sz_mb = os.path.getsize(fpath) / 1e6

    # Read first 5M samples
    raw = np.fromfile(fpath, dtype=np.float32, count=min(5_000_000 * 2, int(sz_mb * 1e6 // 4)))
    if raw.max() < 1e-15:
        # Try complex64
        raw = np.fromfile(fpath, dtype=np.complex64, count=5_000_000)
        iq = raw
        fs = 20e6
    else:
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]
        fs = 100e6 if "100" in fname.lower() else 20e6

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
    log("  [%d] %-30s SNR=%.1f dB  slices=%d" % (imported, tax_id, snr_db, n_slices))

# ── Step 5: Verify ──
log("Step 5: Verify...")
total_classes = len([c for c in os.listdir(CANON) if os.path.isdir(os.path.join(CANON, c))])
total_recs = 0
for c in os.listdir(CANON):
    cp = os.path.join(CANON, c)
    if os.path.isdir(cp):
        total_recs += len([r for r in os.listdir(cp) if os.path.isdir(os.path.join(cp, r))])

# Render a spectrogram from the first imported file
log("Rendering test spectrogram...")
try:
    test_rec = None
    for c in os.listdir(CANON):
        cp = os.path.join(CANON, c)
        if not os.path.isdir(cp): continue
        for r in os.listdir(cp):
            md = os.path.join(cp, r, "metadata.json")
            if os.path.exists(md):
                with open(md) as f: meta = json.load(f)
                if meta.get("source_dataset") == "DroneRFb":
                    test_rec = os.path.join(cp, r)
                    break
        if test_rec: break

    if test_rec:
        s0 = os.path.join(test_rec, "slice_0000.iq")
        cls = os.path.basename(os.path.dirname(test_rec))
        rec = os.path.basename(test_rec)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            generate_images(datapack=s0, file=cls, pack=rec, location=td,
                          fs=20e6, stft_point=256, duration_time=0.1, step=1)
            out_dir = os.path.join(td, cls, rec)
            if os.path.isdir(out_dir):
                jpgs = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
                log("Spectrogram OK: %d frames" % len(jpgs))
            else:
                log("Spectrogram FAILED")
    else:
        log("No DroneRFb recording found for test")
except Exception as e:
    log("Spectrogram test error: %s" % e)

log("=== DONE: %d classes, %d recordings ===" % (total_classes, total_recs))
