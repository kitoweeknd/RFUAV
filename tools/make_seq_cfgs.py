"""Configs for the TODO sequence (all on t256 thumbs, batch 64)."""
import yaml
BASE = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
B = r"E:\Drone_dataset\RFUAV-v2\benchmark"
d0 = yaml.safe_load(open(BASE, encoding="utf-8"))
COMMON = dict(pretrained=True, num_epochs=50, lr=0.0001, batch_size=64,
              grad_accum_steps=1, amp=True, num_workers=4)

def emit(name, model, isz, tree, run, ncls=None, class_dirs=None):
    d = dict(d0); d.update(COMMON)
    d.update(model=model, image_size=isz,
             train=tree + r"\train", val=tree + r"\valid",
             save_path=B + r"\runs" + "\\" + run)
    if ncls:
        d["num_classes"] = ncls
        d["class_names"] = class_dirs
    yaml.dump(d, open(r"D:\ML\RFUAV\configs" + "\\" + name, "w",
                      encoding="utf-8"), allow_unicode=True)
    print("wrote", name)

# (1) 3-seed replicates on the certified grouped t256 tree (46 classes)
for m, isz in [("swinv2_tiny_window8_256", 256), ("vit_base_patch16_224", 224),
               ("mobilenetv3_large_100", 224)]:
    for s in (1, 2, 3):
        emit("sd_%s_s%d.yaml" % (m, s), m, isz,
             B + r"\train_dataset_v1_t256", "sd_%s_s%d" % (m, s))

# (2) SwinV2 x augmentation (2 replicates)
for s in (1, 2):
    emit("ag_swinv2_s%d.yaml" % s, "swinv2_tiny_window8_256", 256,
         B + r"\train_dataset_v1_aug_t256", "ag_swinv2_s%d" % s)

# (3) individual-ID benchmark (18 cls) + model-level (6 cls)
import os
ind = sorted(os.listdir(B + r"\drfb_individual18_t256\train"))
mod = sorted(os.listdir(B + r"\drfb_model6_t256\train"))
emit("id18_swinv2.yaml", "swinv2_tiny_window8_256", 256,
     B + r"\drfb_individual18_t256", "id18_swinv2", len(ind), ind)
emit("id18_resnet50.yaml", "resnet50", 224,
     B + r"\drfb_individual18_t256", "id18_resnet50", len(ind), ind)
emit("id6_swinv2.yaml", "swinv2_tiny_window8_256", 256,
     B + r"\drfb_model6_t256", "id6_swinv2", len(mod), mod)

# (4) cross-source 48-class tier
cs = sorted(os.listdir(B + r"\crosssrc48_t256\train"))
emit("cs_swinv2.yaml", "swinv2_tiny_window8_256", 256,
     B + r"\crosssrc48_t256", "cs_swinv2", len(cs), cs)
