"""Stage 5: materialize ImageFolder train/valid trees from the canonical manifest.

Reads:  E:\Drone_dataset\RFUAV-v2\manifests\canonical_manifest_v1.csv
        E:\Drone_dataset\RFUAV-v2\images_canonical\<tax>\<rec>\*.jpg
Writes: E:\Drone_dataset\RFUAV-v2\train_dataset_v1\{train,valid}\<tax>\*.jpg  (hardlinks)
        E:\Drone_dataset\RFUAV-v2\manifests\train_split_v1_report.json
        D:\ML\RFUAV\configs\train_canonical_v1.yaml

Rules:
  - Only taxonomy classes with >=1 train recording AND >=1 test recording.
  - Non-transmitter classes excluded from the classifier: bg.kong,
    ext.dronerfb.background, unk.weizhi.
  - Split is per-recording (manifest column `split`) -> recording-grouped,
    leakage-clean by construction; verified again here.
  - Hardlink (same NTFS volume), fallback to copy. Idempotent.
"""
import csv, json, os, shutil, sys, glob

ROOT = r"E:\Drone_dataset\RFUAV-v2"
MANIFEST = os.path.join(ROOT, "manifests", "canonical_manifest_v1.csv")
IMAGES = os.path.join(ROOT, "images", "thumbs256", "spectrogram")
OUT = os.path.join(ROOT, "benchmark", "train_dataset_v1_t256")
REPORT = os.path.join(ROOT, "manifests", "train_split_t256_report.json")
CFG_OUT = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"

NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi", "bg.dronerf"}
SPLIT_DIR = {"train": "train", "test": "valid"}

rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
print("manifest rows: %d" % len(rows))

# ---- eligibility: class must have both train and test recordings ----------
by_class = {}
for r in rows:
    by_class.setdefault(r["taxonomy_id"], []).append(r)

eligible, skipped_classes = [], {}
for tax, recs in sorted(by_class.items()):
    splits = set(r["split"] for r in recs)
    if tax in NON_TRANSMITTER:
        skipped_classes[tax] = "non_transmitter (splits: %s)" % ",".join(sorted(splits))
    elif "train" in splits and "test" in splits:
        eligible.append(tax)
    else:
        skipped_classes[tax] = "missing side (splits: %s)" % ",".join(sorted(splits))

print("classes: total=%d eligible=%d skipped=%d"
      % (len(by_class), len(eligible), len(skipped_classes)))

# ---- materialize ----------------------------------------------------------
stats = {}   # tax -> {train_imgs, valid_imgs, train_recs, valid_recs, sources}
linked = copied = existed = missing_img = 0
train_recs_all, valid_recs_all = set(), set()

for tax in eligible:
    st = stats.setdefault(tax, {"train_imgs": 0, "valid_imgs": 0,
                                "train_recs": 0, "valid_recs": 0, "sources": set()})
    for r in by_class[tax]:
        side = SPLIT_DIR.get(r["split"])
        if side is None:
            continue  # excluded / anything else
        rec = r["recording"]
        st["sources"].add(r["source_dataset"])
        st["%s_recs" % side] += 1
        (train_recs_all if side == "train" else valid_recs_all).add(rec)

        src_dir = os.path.join(IMAGES, tax, rec)
        jpgs = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))
        if not jpgs:
            missing_img += 1
            print("WARN no images: %s/%s" % (tax, rec))
            continue
        dst_dir = os.path.join(OUT, side, tax)
        os.makedirs(dst_dir, exist_ok=True)
        for j in jpgs:
            dst = os.path.join(dst_dir, os.path.basename(j))
            if os.path.exists(dst):
                existed += 1
            else:
                try:
                    os.link(j, dst)
                    linked += 1
                except OSError:
                    shutil.copy2(j, dst)
                    copied += 1
            st["%s_imgs" % side] += 1

# ---- leakage + consistency checks -----------------------------------------
straddle = train_recs_all & valid_recs_all
assert not straddle, "LEAKAGE: recordings in both splits: %s" % sorted(straddle)[:5]

train_dirs = set(os.listdir(os.path.join(OUT, "train")))
valid_dirs = set(os.listdir(os.path.join(OUT, "valid")))
assert train_dirs == valid_dirs == set(eligible), (
    "class dir mismatch: train-only=%s valid-only=%s"
    % (sorted(train_dirs - valid_dirs), sorted(valid_dirs - train_dirs)))

n_train = sum(s["train_imgs"] for s in stats.values())
n_valid = sum(s["valid_imgs"] for s in stats.values())
print("images: train=%d valid=%d (linked=%d copied=%d existed=%d missing_rec=%d)"
      % (n_train, n_valid, linked, copied, existed, missing_img))
print("recordings: train=%d valid=%d straddle=0" % (len(train_recs_all), len(valid_recs_all)))

# ---- report ----------------------------------------------------------------
for s in stats.values():
    s["sources"] = sorted(s["sources"])
report = {
    "created_from": "canonical_manifest_v1.csv",
    "out_root": OUT,
    "n_classes": len(eligible),
    "classes": eligible,
    "skipped_classes": skipped_classes,
    "images_train": n_train, "images_valid": n_valid,
    "recordings_train": len(train_recs_all), "recordings_valid": len(valid_recs_all),
    "recording_straddle": 0,
    "per_class": stats,
}
json.dump(report, open(REPORT, "w", encoding="utf-8"), indent=2)
print("report: %s" % REPORT)

print("T256 TREE DONE")
