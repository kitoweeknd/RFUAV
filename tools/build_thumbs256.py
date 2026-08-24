"""Build 256x256 thumbnail trees for all spectrogram images (fast-loader
path). Mirrors images/spectrogram and images/spectrogram_augmented into
images/thumbs256/... JPEG q95. Idempotent."""
import os, glob, time
from PIL import Image

ROOT = r"E:\Drone_dataset\RFUAV-v2"
PAIRS = [(os.path.join(ROOT, "images", "spectrogram"),
          os.path.join(ROOT, "images", "thumbs256", "spectrogram")),
         (os.path.join(ROOT, "images", "spectrogram_augmented"),
          os.path.join(ROOT, "images", "thumbs256", "spectrogram_augmented"))]

def log(m):
    print("%s  %s" % (time.strftime("%H:%M:%S"), m), flush=True)

done = skipped = failed = 0
t0 = time.time()
for src_root, dst_root in PAIRS:
    for j in glob.glob(os.path.join(src_root, "*", "*", "*.jpg")):
        rel = os.path.relpath(j, src_root)
        dst = os.path.join(dst_root, rel)
        if os.path.exists(dst):
            skipped += 1
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im = Image.open(j).convert("RGB").resize((256, 256), Image.BILINEAR)
            im.save(dst, "JPEG", quality=95)
            done += 1
        except Exception as e:
            failed += 1
            log("FAIL %s: %s" % (rel, e))
        if done % 1000 == 0 and done:
            log("progress: %d done (%.0f/min)" % (done, done / max(time.time()-t0, 1) * 60))
log("THUMBS DONE: done=%d skipped=%d failed=%d" % (done, skipped, failed))
if failed > 20:
    raise SystemExit(1)
