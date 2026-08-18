"""Template for future baseline configs: batch 4 + grad-accum 16 (effective 64)
per user directive 2026-08-17 — keeps GPU memory ~2 GB on the shared box.
Usage: python make_baseline_cfg_small.py <src_cfg> <dst_cfg> <run_name> [model]
"""
import sys, yaml

src, dst, run_name = sys.argv[1], sys.argv[2], sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "resnet50"

d = yaml.safe_load(open(src, encoding="utf-8"))
d.update(
    model=model,
    pretrained=True,
    num_epochs=50,
    lr=0.0001,
    batch_size=4,
    grad_accum_steps=16,   # effective batch 4*16=64, comparable with c2-c5
    image_size=224,
    amp=True,
    num_workers=2,
    save_path=r"E:\Drone_dataset\RFUAV-v2\runs" + "\\" + run_name,
)
yaml.dump(d, open(dst, "w", encoding="utf-8"), allow_unicode=True)
print("wrote %s (model=%s batch=4 accum=16 classes=%s)"
      % (dst, model, d["num_classes"]))
