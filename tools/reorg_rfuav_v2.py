"""Reorganize E:\\Drone_dataset\\RFUAV-v2 into three top sections:
rawdata/<source>/<tax>/<rec> (canonical IQ split by origin dataset),
images/{spectrogram, spectrogram_scaled20M, spectrogram_augmented},
benchmark/{train trees, augmented_iq, runs}.
Same-volume moves only (instant renames; hardlinked trees stay valid).
Idempotent: every move checks source existence first. Writes README.md.
"""
import os, json, shutil

ROOT = r"E:\Drone_dataset\RFUAV-v2"
SRC_DIR = {"RFUAV": "RFUAV", "DroneRF": "DroneRF",
           "DroneRFa": "DroneRFa", "DroneRFb-DIR": "DroneRFb"}


def mv(a, b):
    if os.path.exists(a) and not os.path.exists(b):
        os.makedirs(os.path.dirname(b), exist_ok=True)
        shutil.move(a, b)
        print("moved %s -> %s" % (os.path.relpath(a, ROOT), os.path.relpath(b, ROOT)))


# 1) canonical -> rawdata/<source>/<tax>/<rec>
canon = os.path.join(ROOT, "canonical")
raw = os.path.join(ROOT, "rawdata")
n = 0
if os.path.isdir(canon):
    for tax in sorted(os.listdir(canon)):
        tdir = os.path.join(canon, tax)
        if not os.path.isdir(tdir):
            continue
        for rec in sorted(os.listdir(tdir)):
            rdir = os.path.join(tdir, rec)
            mp = os.path.join(rdir, "metadata.json")
            if not os.path.isdir(rdir) or not os.path.exists(mp):
                print("SKIP (no metadata): %s/%s" % (tax, rec))
                continue
            src = json.load(open(mp, encoding="utf-8")).get("source_dataset", "RFUAV")
            dst = os.path.join(raw, SRC_DIR.get(src, src), tax, rec)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(rdir, dst)
                n += 1
        if not os.listdir(tdir):
            os.rmdir(tdir)
    if os.path.isdir(canon) and not os.listdir(canon):
        os.rmdir(canon)
print("rawdata: %d recordings placed by source" % n)

# 2) images
mv(os.path.join(ROOT, "images_canonical"), os.path.join(ROOT, "images", "spectrogram"))
mv(os.path.join(ROOT, "images_scaled"), os.path.join(ROOT, "images", "spectrogram_scaled20M"))
mv(os.path.join(ROOT, "augmented", "images"), os.path.join(ROOT, "images", "spectrogram_augmented"))

# 3) benchmark
for t in ("train_dataset_v1", "train_dataset_v1_aug", "train_dataset_v2scaled", "runs"):
    mv(os.path.join(ROOT, t), os.path.join(ROOT, "benchmark", t))
mv(os.path.join(ROOT, "augmented", "iq"), os.path.join(ROOT, "benchmark", "augmented_iq"))
aug = os.path.join(ROOT, "augmented")
if os.path.isdir(aug) and not os.listdir(aug):
    os.rmdir(aug)
stg = os.path.join(ROOT, "staging")
if os.path.isdir(stg):
    empty = all(not fs for _, _, fs in os.walk(stg))
    if empty:
        shutil.rmtree(stg)
        print("removed empty staging/")

open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write("""# RFUAV-v2 Layout (since 2026-08-20)

- `rawdata/<source>/<taxonomy>/<recording>/` - canonical IQ (float32
  interleaved, slice_%04d.iq + metadata.json). Sources: RFUAV (native,
  100 MHz), DroneRF (20 MHz, hilbert+decimated), DroneRFa (20 MHz),
  DroneRFb (100 MHz).
- `images/spectrogram/<tax>/<rec>/` - one spectrogram per slice (native
  full-band render, the benchmark default).
- `images/spectrogram_scaled20M/` - fixed-20MHz-window ablation renders.
- `images/spectrogram_augmented/` - channel-augmented renders (+ per-slice
  aug_params.jsonl).
- `benchmark/train_dataset_v1{,_aug,...}/` - ImageFolder train/valid trees
  (hardlinks into images/), `benchmark/augmented_iq/` - augmented raw IQ
  (regenerable bit-exact from seeds+params), `benchmark/runs/` - weights.
- `manifests/` - all accounting (canonical_manifest_v1.csv is the master).
- `quarantine/` - never-delete holding area (orphans, format twins).
Rollback contract: this whole tree is the writable area; originals under
E:\\Drone_dataset\\{RFUAV,RF,RFA,DroneRFb} are READ-ONLY.
""")
print("README written. REORG DONE")
