"""Run remaining Exp B models, skip existing."""
import torch, timm, os, subprocess, sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

py = r"C:\ML\anaconda\envs\p0\python.exe"
train = r"D:\ML\RFUAV\train.py"
cfg_dir = r"D:\ML\RFUAV\configs"
val_path = r"E:\Drone_dataset\RFUAV-v2\split_23\val"

models = [
    ("resnet101", "resnet101", 640),
    ("resnet152", "resnet152", 640),
    ("vit_b_32", "vit_base_patch32_224", 224),
    ("vit_l_16", "vit_large_patch16_224", 224),
    ("vit_l_32", "vit_large_patch32_224", 224),
    ("swin_t", "swinv2_tiny_window8_256", 256),
    ("swin_s", "swinv2_small_window8_256", 256),
    ("swin_b", "swinv2_base_window8_256", 256),
    ("eff_b0", "tf_efficientnet_b0", 224),
    ("eff_b1", "tf_efficientnet_b1", 224),
    ("eff_b2", "tf_efficientnet_b2", 224),
    ("eff_b3", "tf_efficientnet_b3", 224),
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

results = []
for cfg_name, model_name, img_size in models:
    ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % cfg_name
    if os.path.exists(ckpt):
        print("SKIP %s (exists)" % cfg_name)
        # Still evaluate
        m = timm.create_model(model_name, pretrained=False, num_classes=23)
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        m.load_state_dict(state, strict=False)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        m.cuda().eval()
        ds = datasets.ImageFolder(val_path, transform=transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor()]))
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        c = t = 0
        with torch.no_grad():
            for imgs, labels in dl:
                c += (m(imgs.cuda()).argmax(1) == labels.cuda()).sum().item()
                t += labels.size(0)
        acc = 100 * c / t
        print("  ACC: %.2f%%  PARAMS: %.1fM" % (acc, params))
        del m, state
        continue

    cfg_path = os.path.join(cfg_dir, "expB_%s.yaml" % cfg_name)
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
    with open(cfg_path, "w") as f:
        f.write(yaml)

    print(">>> %s (%s) <<<" % (cfg_name, model_name))
    ret = subprocess.run([py, train, "--cfg", cfg_path], check=False)
    if ret.returncode != 0:
        print("FAILED: %s (exit %d)" % (cfg_name, ret.returncode))
        continue

    m = timm.create_model(model_name, pretrained=False, num_classes=23)
    state = torch.load(ckpt, map_location="cuda", weights_only=True)
    m.load_state_dict(state, strict=False)
    params = sum(p.numel() for p in m.parameters()) / 1e6
    m.cuda().eval()
    ds = datasets.ImageFolder(val_path, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor()]))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    c = t = 0
    with torch.no_grad():
        for imgs, labels in dl:
            c += (m(imgs.cuda()).argmax(1) == labels.cuda()).sum().item()
            t += labels.size(0)
    acc = 100 * c / t
    print("  ACC: %.2f%%  PARAMS: %.1fM" % (acc, params))
    del m, state

# Write final results
out = r"D:\ML\RFUAV\tools\expB_scaling_results.txt"
all_models = [
    ("resnet18", 640), ("resnet34", 640), ("resnet50", 640), ("resnet101", 640), ("resnet152", 640),
    ("vit_b_16", 224), ("vit_b_32", 224), ("vit_l_16", 224), ("vit_l_32", 224),
    ("swin_t", 256), ("swin_s", 256), ("swin_b", 256),
    ("eff_b0", 224), ("eff_b1", 224), ("eff_b2", 224), ("eff_b3", 224),
]
with open(out, "w") as f:
    f.write("%-20s %10s %8s\n" % ("MODEL", "PARAMS_M", "ACC"))
    f.write("-" * 40 + "\n")
    for name, sz in all_models:
        ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % name
        if not os.path.exists(ckpt):
            f.write("%-20s %10s %8s\n" % (name, "-", "MISSING"))
            continue
        m = timm.create_model(name, pretrained=False, num_classes=23)
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        m.load_state_dict(state, strict=False)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        m.cuda().eval()
        ds = datasets.ImageFolder(val_path, transform=transforms.Compose([
            transforms.Resize((sz, sz)), transforms.ToTensor()]))
        dl = DataLoader(ds, batch_size=64, shuffle=False)
        c = t = 0
        with torch.no_grad():
            for imgs, labels in dl:
                c += (m(imgs.cuda()).argmax(1) == labels.cuda()).sum().item()
                t += labels.size(0)
        acc = 100 * c / t
        f.write("%-20s %10.1f %8.2f%%\n" % (name, params, acc))
        del m, state
        print("%-20s %10.1f %8.2f%%" % (name, params, acc))

print("Results written to", out)
