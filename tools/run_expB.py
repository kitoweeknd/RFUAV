"""Generate Exp B configs and run 7 models (batch_size=32)."""
import os, subprocess, sys

py = r"C:\ML\anaconda\envs\p0\python.exe"
train = r"D:\ML\RFUAV\train.py"
cfg_dir = r"D:\ML\RFUAV\configs"

models = [
    ("resnet18", "resnet18", 640),
    ("vit_base_patch16_224", "vit_base_patch16_224", 224),
    ("swinv2_tiny_window8_256", "swinv2_tiny_window8_256", 256),
    ("mobilenetv3_small_100", "mobilenetv3_small_100", 224),
    ("convnext_tiny", "convnext_tiny", 224),
    ("tf_efficientnetv2_s", "tf_efficientnetv2_s", 224),
    ("densenet121", "densenet121", 224),
]

class_names = [
    "rc.devention.devo_10", "rc.flysky.el18", "rc.flysky.fs_i6x", "rc.flysky.nv14",
    "rc.frsky.x9dp2019", "rc.futaba.t10j", "rc.futaba.t16iz", "rc.futaba.t18sz",
    "rc.jrpropo.xg14", "rc.radiolink.at10_ii", "rc.siyi.ft24", "rc.siyi.mk15",
    "rc.siyi.mk32", "rc.skydroid.t10", "rc.wfly.et10", "rc.yunzhuo.h16",
    "uav.autel.evo_nano", "uav.dji.avata2", "uav.dji.fpv_combo",
    "uav.dji.mavic3_pro", "uav.dji.mini3", "uav.dji.mini4_pro",
    "uav.parrot.parrot_unspecified",
]

for cfg_name, model_name, img_size in models:
    yaml = """train: E:\\Drone_dataset\\RFUAV-v2\\split_23\\train
val: E:\\Drone_dataset\\RFUAV-v2\\split_23\\val
save_path: E:\\Drone_dataset\\RFUAV-v2\\checkpoints\\expB_%s
model: %s
pretrained: True
weights: None
batch_size: 32
image_size: %d
num_classes: 23
class_names: %s
device: cuda
num_epochs: 50
shuffle: True
lr: 0.0001
""" % (cfg_name, model_name, img_size, class_names)
    cfg_path = os.path.join(cfg_dir, "expB_%s.yaml" % cfg_name)
    with open(cfg_path, "w") as f:
        f.write(yaml)

print("=== Running Exp B: 7 models, 23 classes, batch=32 ===")
for cfg_name, model_name, img_size in models:
    cfg_path = os.path.join(cfg_dir, "expB_%s.yaml" % cfg_name)
    print("\n>>> %s <<<" % model_name)
    ret = subprocess.run([py, train, "--cfg", cfg_path], check=False)
    if ret.returncode != 0:
        print("FAILED: %s (exit %d)" % (model_name, ret.returncode))
    else:
        print("DONE: %s" % model_name)
