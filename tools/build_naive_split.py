"""Build the NAIVE (slice-level random) split arm for the naive-vs-grouped
headline comparison.

Same image pool and same per-class test counts as the grouped split
(benchmark/train_dataset_v1) — the ONLY difference is that test images are
drawn at random slice level (seed 42), so slices of one recording land on
both sides, exactly like the original RFUAV protocol. Also measures the
leakage this creates: fraction of valid images whose source recording also
appears in train.
"""
import os, glob, json, random, re
from collections import defaultdict

ROOT = r"E:\Drone_dataset\RFUAV-v2"
SRC = os.path.join(ROOT, "benchmark", "train_dataset_v1")
OUT = os.path.join(ROOT, "benchmark", "train_dataset_naive")
REPORT = os.path.join(ROOT, "manifests", "naive_split_report.json")

random.seed(42)


def rec_of(fname):
    """source recording = image name minus the _sNNNN.jpg suffix."""
    return re.sub(r"_s\d{4}\.jpg$", "", fname)


per_class = {}
straddle_imgs = total_valid = 0
train_recs_count = defaultdict(set)
for tax in sorted(os.listdir(os.path.join(SRC, "train"))):
    pool = []
    for side in ("train", "valid"):
        for j in glob.glob(os.path.join(SRC, side, tax, "*.jpg")):
            pool.append(os.path.basename(j))
    n_test = len(glob.glob(os.path.join(SRC, "valid", tax, "*.jpg")))
    pool_paths = {}
    for side in ("train", "valid"):
        for j in glob.glob(os.path.join(SRC, side, tax, "*.jpg")):
            pool_paths[os.path.basename(j)] = j
    test_set = set(random.sample(sorted(pool), n_test))
    for name in sorted(pool):
        side = "valid" if name in test_set else "train"
        dst_dir = os.path.join(OUT, side, tax)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(dst):
            os.link(pool_paths[name], dst)
        if side == "train":
            train_recs_count[tax].add(rec_of(name))
    # leakage measurement
    for name in test_set:
        total_valid += 1
        if rec_of(name) in train_recs_count[tax]:
            straddle_imgs += 1
    per_class[tax] = {"pool": len(pool), "test": n_test}

n_train = sum(len(glob.glob(os.path.join(OUT, "train", t, "*.jpg")))
              for t in os.listdir(os.path.join(OUT, "train")))
n_valid = sum(len(glob.glob(os.path.join(OUT, "valid", t, "*.jpg")))
              for t in os.listdir(os.path.join(OUT, "valid")))
leak_pct = 100.0 * straddle_imgs / max(total_valid, 1)
print("naive split: classes=%d train=%d valid=%d" % (len(per_class), n_train, n_valid))
print("LEAKAGE: %d/%d valid images (%.1f%%) share a recording with train"
      % (straddle_imgs, total_valid, leak_pct))
json.dump({"seed": 42, "classes": len(per_class), "train": n_train,
           "valid": n_valid, "leak_valid_images_pct": round(leak_pct, 2),
           "per_class": per_class},
          open(REPORT, "w", encoding="utf-8"), indent=2)
print("report:", REPORT)
