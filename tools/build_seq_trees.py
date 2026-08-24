"""Build the remaining fast-lane (t256) trees for the TODO sequence:
1) train_dataset_v1_aug_t256  - grouped train + augmented thumbs, valid untouched
2) drfb_individual18_t256     - DroneRFb 18 individual classes (IN=train, OUT=valid)
3) drfb_model6_t256           - same data collapsed to 6 anonymous models
4) crosssrc48_t256            - 46 classes + t14sg & phantom3 made evaluable by
   moving their foreign-source sessions to valid (cross-source tier);
   master manifest untouched.
All hardlinks from images/thumbs256. Idempotent."""
import os, csv, glob

ROOT = r"E:\Drone_dataset\RFUAV-v2"
TH = os.path.join(ROOT, "images", "thumbs256", "spectrogram")
TH_AUG = os.path.join(ROOT, "images", "thumbs256", "spectrogram_augmented")
T256 = os.path.join(ROOT, "benchmark", "train_dataset_v1_t256")
MANIFEST = os.path.join(ROOT, "manifests", "canonical_manifest_v1.csv")
NON_TRANSMITTER = {"bg.kong", "ext.dronerfb.background", "unk.weizhi", "bg.dronerf"}

rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))


def link_all(src_dir, dst_dir):
    n = 0
    os.makedirs(dst_dir, exist_ok=True)
    for j in glob.glob(os.path.join(src_dir, "*.jpg")):
        d = os.path.join(dst_dir, os.path.basename(j))
        if not os.path.exists(d):
            os.link(j, d)
            n += 1
    return n


# ---- 1) aug tree -----------------------------------------------------------
OUT1 = os.path.join(ROOT, "benchmark", "train_dataset_v1_aug_t256")
n = 0
for side in ("train", "valid"):
    for tax in os.listdir(os.path.join(T256, side)):
        n += link_all(os.path.join(T256, side, tax), os.path.join(OUT1, side, tax))
for tax in os.listdir(TH_AUG):
    ddir = os.path.join(OUT1, "train", tax)
    if not os.path.isdir(ddir):
        continue  # augmented class not in current eligible set
    for rec in os.listdir(os.path.join(TH_AUG, tax)):
        n += link_all(os.path.join(TH_AUG, tax, rec), ddir)
print("aug_t256 linked=%d train=%d valid=%d" % (
    n, len(glob.glob(os.path.join(OUT1, "train", "*", "*.jpg"))),
    len(glob.glob(os.path.join(OUT1, "valid", "*", "*.jpg")))))

# ---- 2/3) DroneRFb individual + model trees --------------------------------
OUT2 = os.path.join(ROOT, "benchmark", "drfb_individual18_t256")
OUT3 = os.path.join(ROOT, "benchmark", "drfb_model6_t256")
n2 = n3 = 0
for r in rows:
    tax = r["taxonomy_id"]
    if not (tax.startswith("ext.dronerfb.") and tax[-1].isdigit()):
        continue
    side = {"train": "train", "test": "valid"}.get(r["split"])
    if side is None:
        continue
    src = os.path.join(TH, tax, r["recording"])
    n2 += link_all(src, os.path.join(OUT2, side, tax))
    n3 += link_all(src, os.path.join(OUT3, side, "model_" + tax.split(".")[-1][0]))
print("drfb trees: individual18 linked=%d model6 linked=%d" % (n2, n3))

# ---- 4) cross-source 48-class tree -----------------------------------------
OUT4 = os.path.join(ROOT, "benchmark", "crosssrc48_t256")
CROSS_TEST = {("rc.futaba.t14sg", "DroneRFa"), ("uav.dji.phantom3", "DroneRF")}
by_class = {}
for r in rows:
    by_class.setdefault(r["taxonomy_id"], set()).add(r["split"])
eligible = {t for t, s in by_class.items()
            if t not in NON_TRANSMITTER and "train" in s and "test" in s}
eligible |= {"rc.futaba.t14sg", "uav.dji.phantom3"}
n4 = 0
tr_recs, va_recs = set(), set()
for r in rows:
    tax = r["taxonomy_id"]
    if tax not in eligible:
        continue
    split = r["split"]
    if (tax, r["source_dataset"]) in CROSS_TEST:
        split = "test"
    side = {"train": "train", "test": "valid"}.get(split)
    if side is None:
        continue
    (tr_recs if side == "train" else va_recs).add(r["recording"])
    n4 += link_all(os.path.join(TH, tax, r["recording"]), os.path.join(OUT4, side, tax))
assert not (tr_recs & va_recs), "crosssrc straddle!"
td = set(os.listdir(os.path.join(OUT4, "train")))
vd = set(os.listdir(os.path.join(OUT4, "valid")))
assert td == vd == eligible, "crosssrc class mismatch: %s / %s" % (
    sorted(td - vd), sorted(vd - td))
print("crosssrc48 classes=%d linked=%d straddle=0" % (len(eligible), n4))
print("SEQ TREES DONE")
