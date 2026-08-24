"""Configs for the naive-split arm: batch 64 (large batch re-allowed
2026-08-24), otherwise the uniform recipe."""
import yaml
BASE = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
d0 = yaml.safe_load(open(BASE, encoding="utf-8"))
for name, isz in [("resnet50", 224), ("swinv2_tiny_window8_256", 256)]:
    d = dict(d0)
    d.update(model=name, pretrained=True, num_epochs=50, lr=0.0001,
             batch_size=64, grad_accum_steps=1, image_size=isz, amp=True,
             num_workers=4,
             train=r"E:\Drone_dataset\RFUAV-v2\benchmark\train_dataset_naive\train",
             val=r"E:\Drone_dataset\RFUAV-v2\benchmark\train_dataset_naive\valid",
             save_path=r"E:\Drone_dataset\RFUAV-v2\benchmark\runs\nv_" + name)
    p = r"D:\ML\RFUAV\configs\nv_" + name + ".yaml"
    yaml.dump(d, open(p, "w", encoding="utf-8"), allow_unicode=True)
    print("wrote", p)
