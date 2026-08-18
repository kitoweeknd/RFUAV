"""Set the c5 config to batch 2 + grad-accum 32 (effective 64)."""
import yaml
P = r"D:\ML\RFUAV\configs\exp_c5_resnet50_aug_v1.yaml"
d = yaml.safe_load(open(P, encoding="utf-8"))
d["batch_size"] = 2
d["grad_accum_steps"] = 32
d["num_workers"] = 2
yaml.dump(d, open(P, "w", encoding="utf-8"), allow_unicode=True)
print("patched c5: batch=2 accum=32 workers=2")
