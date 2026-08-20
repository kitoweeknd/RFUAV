"""Build images_scaled: every slice rendered at a FIXED 20 MHz frequency span.

- 20 MHz sources (DroneRFa, DroneRF): hardlink the existing canonical images
  (they already are 20 MHz full-band renders with the same STFT params).
- 100 MHz sources (RFUAV, DroneRFb): per slice, pick the 20 MHz sub-band with
  maximum energy via FFT (equivalent to a receiver tuned to the active band),
  extract those bins, IFFT back to 2M complex samples @ 20 MHz, render.
  Window center offsets recorded per recording in window_offsets.jsonl.
Idempotent (skip-if-exists). IQ temps deleted after each render.
"""
import os, sys, json, glob, time
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
from graphic.RawDataProcessor import generate_images

ROOT = r"E:\Drone_dataset\RFUAV-v2"
CANON = os.path.join(ROOT, "canonical")
IMAGES = os.path.join(ROOT, "images_canonical")
OUT = os.path.join(ROOT, "images_scaled")
SPAN = 20e6


def log(msg):
    print("%s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def best_window(X_shifted, wbins):
    """start index of the max-energy contiguous window of wbins bins."""
    p = np.abs(X_shifted) ** 2
    c = np.cumsum(p)
    ssum = c[wbins - 1:] - np.concatenate(([0.0], c[:-wbins]))
    return int(np.argmax(ssum))


jobs_link = jobs_render = skipped = failed = 0
t0 = time.time()
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
        fs = float(md.get("sample_rate_hz") or 100e6)
        odir = os.path.join(OUT, tax, rec)
        os.makedirs(odir, exist_ok=True)
        slices = sorted(glob.glob(os.path.join(rdir, "slice_*.iq")))

        if fs <= SPAN + 1:  # already 20 MHz: hardlink existing renders
            for spath in slices:
                si = int(os.path.basename(spath)[6:10])
                src_img = os.path.join(IMAGES, tax, rec, "%s_s%04d.jpg" % (rec, si))
                dst_img = os.path.join(odir, "%s_s%04d.jpg" % (rec, si))
                if os.path.exists(dst_img):
                    skipped += 1
                elif os.path.exists(src_img):
                    os.link(src_img, dst_img)
                    jobs_link += 1
                else:
                    failed += 1
                    log("MISSING SRC IMG %s/%s s%04d" % (tax, rec, si))
            continue

        # 100 MHz source: window + decimate via FFT bin extraction
        ratio = int(round(fs / SPAN))          # 5
        wlog = open(os.path.join(odir, "window_offsets.jsonl"), "a", encoding="utf-8")
        for spath in slices:
            si = int(os.path.basename(spath)[6:10])
            dst_img = os.path.join(odir, "%s_s%04d.jpg" % (rec, si))
            if os.path.exists(dst_img):
                skipped += 1
                continue
            try:
                raw = np.fromfile(spath, dtype=np.float32)
                n = raw.size // 2
                x = np.empty(n, dtype=np.complex64)
                x.real = raw[0::2]; x.imag = raw[1::2]
                del raw
                X = np.fft.fftshift(np.fft.fft(x))
                del x
                wbins = n // ratio
                s = best_window(X, wbins)
                center_off = (s + wbins / 2 - n / 2) / n * fs
                Xw = np.fft.ifftshift(X[s:s + wbins])
                del X
                y = np.fft.ifft(Xw).astype(np.complex64)
                del Xw
                tmp = os.path.join(odir, "_tmp.iq")
                buf = np.empty(2 * len(y), dtype=np.float32)
                buf[0::2] = y.real; buf[1::2] = y.imag
                buf.tofile(tmp)
                ny = len(y)
                del y, buf
                imgs = generate_images(datapack=tmp, fs=int(SPAN), stft_point=1024,
                                       duration_time=min(0.1, ny / SPAN),
                                       location='buffer', file_type=np.float32)
                os.remove(tmp)
                if not imgs:
                    failed += 1
                    continue
                imgs[0].convert("RGB").save(dst_img, "JPEG", quality=95)
                wlog.write(json.dumps({"slice": si,
                                       "center_offset_hz": round(center_off)}) + "\n")
                jobs_render += 1
            except Exception as e:
                failed += 1
                log("FAIL %s/%s s%04d: %s" % (tax, rec, si, e))
            if (jobs_render + failed) % 200 == 0 and (jobs_render + failed) > 0:
                log("progress: rendered=%d linked=%d skipped=%d failed=%d (%.1f/min)"
                    % (jobs_render, jobs_link, skipped, failed,
                       jobs_render / max(time.time() - t0, 1) * 60))
        wlog.close()

log("SCALED RENDER DONE: rendered=%d linked=%d skipped=%d failed=%d"
    % (jobs_render, jobs_link, skipped, failed))
if failed > 60:
    sys.exit(1)
