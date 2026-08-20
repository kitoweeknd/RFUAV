"""Two configs for the rendering-scale ablation (identical recipe, batch 4+accum 16)."""
import yaml
BASE = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
d0 = yaml.safe_load(open(BASE, encoding="utf-8"))
common = dict(model="resnet50", pretrained=True, num_epochs=50, lr=0.0001,
              batch_size=4, grad_accum_steps=16, image_size=224, amp=True,
              num_workers=2)
for name, train_root, run in [
    ("exp_c8a_native46.yaml", r"E:\Drone_dataset\RFUAV-v2\train_dataset_v1", "c8a_resnet50_native46"),
    ("exp_c8b_scaled46.yaml", r"E:\Drone_dataset\RFUAV-v2\train_dataset_v2scaled", "c8b_resnet50_scaled46"),
]:
    d = dict(d0); d.update(common)
    d["train"] = train_root + r"\train"
    d["val"] = train_root + r"\valid"
    d["save_path"] = r"E:\Drone_dataset\RFUAV-v2\runs" + "\\" + run
    yaml.dump(d, open(r"D:\ML\RFUAV\configs" + "\\" + name, "w", encoding="utf-8"), allow_unicode=True)
    print("wrote", name, "classes=", d["num_classes"])
