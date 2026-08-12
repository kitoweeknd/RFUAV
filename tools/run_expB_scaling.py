"""Exp B scaling: 4 model series, multiple sizes each."""
import torch, timm, os, subprocess, sys

py = r"C:\ML\anaconda\envs\p0\python.exe"
train = r"D:\ML\RFUAV\train.py"
cfg_dir = r"D:\ML\RFUAV\configs"
val_path = r"E:\Drone_dataset\RFUAV-v2\split_23\val"

models = [
    # ResNet series
    ("resnet18", "resnet18", 640),
    ("resnet34", "resnet34", 640),
    ("resnet50", "resnet50", 640),
    ("resnet101", "resnet101", 640),
    ("resnet152", "resnet152", 640),
    # ViT series
    ("vit_b_16", "vit_base_patch16_224", 224),
    ("vit_b_32", "vit_base_patch32_224", 224),
    ("vit_l_16", "vit_large_patch16_224", 224),
    ("vit_l_32", "vit_large_patch32_224", 224),
    # Swin series
    ("swin_t", "swinv2_tiny_window8_256", 256),
    ("swin_s", "swinv2_small_window8_256", 256),
    ("swin_b", "swinv2_base_window8_256", 256),
    # EfficientNet series
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
        print("SKIP %s (checkpoint exists)" % cfg_name)
        # Evaluate
        m = timm.create_model(model_name, pretrained=False, num_classes=23)
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        m.load_state_dict(state, strict=False)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        del m, state
        results.append((cfg_name, model_name, params, "SKIP"))
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

    print(">>> %s (%s, %d px) <<<" % (cfg_name, model_name, img_size))
    ret = subprocess.run([py, train, "--cfg", cfg_path], check=False)
    if ret.returncode != 0:
        print("FAILED: %s (exit %d)" % (cfg_name, ret.returncode))
        results.append((cfg_name, model_name, -1, "FAIL"))
    else:
        # Evaluate
        ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % cfg_name
        m = timm.create_model(model_name, pretrained=False, num_classes=23)
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        m.load_state_dict(state, strict=False)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
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
        results.append((cfg_name, model_name, params, acc))

print("\n=== RESULTS ===")
print("%-15s %-25s %8s %8s" % ("CFG", "MODEL", "PARAMS_M", "ACC"))
for cfg, model, params, acc in results:
    print("%-15s %-25s %8.1f %8.2f%%" % (cfg, model, params, acc if isinstance(acc, float) else 0))
