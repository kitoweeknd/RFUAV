"""Evaluate all 7 Exp B models on all SNR levels."""
import torch, timm, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

models = [
    ("resnet18", 640), ("vit_base_patch16_224", 224),
    ("swinv2_tiny_window8_256", 256), ("mobilenetv3_small_100", 224),
    ("convnext_tiny", 224), ("tf_efficientnetv2_s", 224), ("densenet121", 224),
]
base = r"E:\Drone_dataset\RFUAV-v2\split_snr"
out = r"D:\ML\RFUAV\tools\snr_results.txt"

with open(out, "w") as fh:
    fh.write("%-6s " % "SNR")
    for name, _ in models: fh.write("%8s " % name[:8])
    fh.write("\n")
    
    for snr_dir in sorted(os.listdir(base)):
        val_path = os.path.join(base, snr_dir)
        if not os.path.isdir(val_path): continue
        
        fh.write("%-6s " % snr_dir)
        for name, img_size in models:
            ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % name
            model = timm.create_model(name, pretrained=False, num_classes=23)
            state = torch.load(ckpt, map_location="cuda", weights_only=True)
            model.load_state_dict(state, strict=False)
            model.cuda().eval()
            ds = datasets.ImageFolder(val_path, transform=transforms.Compose([
                transforms.Resize((img_size, img_size)), transforms.ToTensor()]))
            dl = DataLoader(ds, batch_size=8, shuffle=False)
            c = t = 0
            with torch.no_grad():
                for imgs, labels in dl:
                    c += (model(imgs.cuda()).argmax(1) == labels.cuda()).sum().item()
                    t += labels.size(0)
            acc = 100 * c / t if t > 0 else 0
            fh.write("%7.1f%% " % acc)
            del model, state
        fh.write("\n")
        fh.flush()
        print("%-6s done" % snr_dir)
print("->", out)
