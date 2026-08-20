"""Build train_dataset_v1_aug: train = original train images + augmented
images (hardlinks); valid = original valid images unchanged. Writes the c5
config from the current smoke config's class list."""
import os, glob, yaml

ROOT = r"E:\Drone_dataset\RFUAV-v2"
SRC = os.path.join(ROOT, "benchmark", "train_dataset_v1")
AUG_IMG = os.path.join(ROOT, "images", "spectrogram_augmented")
OUT = os.path.join(ROOT, "benchmark", "train_dataset_v1_aug")
CFG_SRC = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
CFG_DST = r"D:\ML\RFUAV\configs\exp_c7_resnet50_aug46.yaml"

linked = 0
for side in ("train", "valid"):
    for tax in sorted(os.listdir(os.path.join(SRC, side))):
        sdir = os.path.join(SRC, side, tax)
        ddir = os.path.join(OUT, side, tax)
        os.makedirs(ddir, exist_ok=True)
        for j in glob.glob(os.path.join(sdir, "*.jpg")):
            dst = os.path.join(ddir, os.path.basename(j))
            if not os.path.exists(dst):
                os.link(j, dst)
                linked += 1

aug_linked = 0
for tax in sorted(os.listdir(AUG_IMG)):
    ddir = os.path.join(OUT, "train", tax)
    if not os.path.isdir(ddir):
        raise SystemExit("ABORT: augmented class %s not in train tree" % tax)
    for rec in sorted(os.listdir(os.path.join(AUG_IMG, tax))):
        for j in glob.glob(os.path.join(AUG_IMG, tax, rec, "*.jpg")):
            dst = os.path.join(ddir, os.path.basename(j))
            if not os.path.exists(dst):
                os.link(j, dst)
                aug_linked += 1

train_dirs = set(os.listdir(os.path.join(OUT, "train")))
valid_dirs = set(os.listdir(os.path.join(OUT, "valid")))
assert train_dirs == valid_dirs, "class dir mismatch in aug tree"
n_train = sum(len(glob.glob(os.path.join(OUT, "train", t, "*.jpg"))) for t in train_dirs)
n_valid = sum(len(glob.glob(os.path.join(OUT, "valid", t, "*.jpg"))) for t in valid_dirs)
print("aug tree: classes=%d train=%d (aug_new=%d) valid=%d base_linked=%d"
      % (len(train_dirs), n_train, aug_linked, n_valid, linked))

d = yaml.safe_load(open(CFG_SRC, encoding="utf-8"))
d.update(
    model="resnet50", pretrained=True, num_epochs=50, lr=0.0001,
    batch_size=4, grad_accum_steps=16, image_size=224, amp=True, num_workers=2,
    train=os.path.join(OUT, "train"),
    val=os.path.join(OUT, "valid"),
    save_path=os.path.join(ROOT, "benchmark", "runs", "c7_resnet50_aug46"),
)
yaml.dump(d, open(CFG_DST, "w", encoding="utf-8"), allow_unicode=True)
print("config: %s (classes=%d)" % (CFG_DST, d["num_classes"]))
