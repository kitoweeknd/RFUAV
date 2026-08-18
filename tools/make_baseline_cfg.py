"""Derive the c1 baseline config from the smoke config (keeps the 61-class list)."""
import yaml

SRC = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
DST = r"D:\ML\RFUAV\configs\exp_c1_resnet50_canonical_v1.yaml"

d = yaml.safe_load(open(SRC, encoding="utf-8"))
d.update(
    model="resnet50",
    pretrained=True,
    num_epochs=50,
    lr=0.0001,
    batch_size=64,
    image_size=224,
    amp=True,
    num_workers=4,
    save_path=r"E:\Drone_dataset\RFUAV-v2\runs\c1_resnet50_canonical_v1",
)
yaml.dump(d, open(DST, "w", encoding="utf-8"), allow_unicode=True)
print("wrote %s (model=%s epochs=%s classes=%s)"
      % (DST, d["model"], d["num_epochs"], d["num_classes"]))
