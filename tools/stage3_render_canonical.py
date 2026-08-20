"""Stage 3: render every canonical slice into the unified spectrogram tree.

Output: E:\Drone_dataset\RFUAV-v2\images_canonical\<tax>\<rec>\<rec>_sNNNN.jpg
One image per slice, fs taken from metadata.json (100 MHz RFUAV/DroneRFb,
20 MHz DroneRFa), same render path as the training images (STFT 1024, jet,
1920x1440 JPEG q95, fixed post-log10 clip).
"""
import os, re, json, time, sys
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
from graphic.RawDataProcessor import generate_images

RAW = r"E:\Drone_dataset\RFUAV-v2\rawdata"
OUT = r"E:\Drone_dataset\RFUAV-v2\images\spectrogram"
SLICE_RE = re.compile(r"^slice_(\d{4})\.iq$")

def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

import glob as _glob
jobs = []
for mpath in sorted(_glob.glob(os.path.join(RAW, "*", "*", "*", "metadata.json"))):
    rdir = os.path.dirname(mpath)
    rec = os.path.basename(rdir)
    tax = os.path.basename(os.path.dirname(rdir))
    fs = 100e6
    try:
        with open(mpath) as f:
            fs = float(json.load(f).get("sample_rate_hz") or 100e6)
    except Exception:
        pass
    for f in sorted(os.listdir(rdir)):
        m = SLICE_RE.match(f)
        if m:
            jobs.append((tax, rec, os.path.join(rdir, f), int(m.group(1)), fs))

log("Stage 3 start: %d slices to render" % len(jobs))

done = skipped = failed = 0
for tax, rec, spath, sidx, fs in jobs:
    out_dir = os.path.join(OUT, tax, rec)
    out_path = os.path.join(out_dir, "%s_s%04d.jpg" % (rec, sidx))
    if os.path.exists(out_path):
        skipped += 1
        continue
    n = os.path.getsize(spath) // 8
    if n < 1024:
        failed += 1
        continue
    duration = min(0.1, n / fs)
    try:
        imgs = generate_images(datapack=spath, fs=int(fs), stft_point=1024,
                               duration_time=duration, location='buffer',
                               file_type=np.float32)
        if not imgs:
            failed += 1
            log("  NO FRAME %s/%s s%04d" % (tax, rec, sidx))
            continue
        os.makedirs(out_dir, exist_ok=True)
        imgs[0].convert("RGB").save(out_path, "JPEG", quality=95)
        done += 1
    except Exception as e:
        failed += 1
        log("  RENDER FAIL %s/%s s%04d: %s" % (tax, rec, sidx, e))
    if (done + failed) % 100 == 0:
        log("  progress: done=%d skipped=%d failed=%d / %d" % (done, skipped, failed, len(jobs)))

log("Stage 3 DONE: rendered=%d skipped=%d failed=%d" % (done, skipped, failed))
if failed > len(jobs) * 0.1:
    sys.exit(1)
