"""Reduce loader workers for c5 (32GB box, big JPEGs); relaunch-safe."""
import yaml

P = r"D:\ML\RFUAV\configs\exp_c5_resnet50_aug_v1.yaml"
d = yaml.safe_load(open(P, encoding="utf-8"))
d["num_workers"] = 2
d["prefetch_factor"] = 2
yaml.dump(d, open(P, "w", encoding="utf-8"), allow_unicode=True)
print("patched: num_workers=2 (classes=%d, train=%s)" % (d["num_classes"], d["train"]))
