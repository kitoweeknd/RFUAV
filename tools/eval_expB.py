import torch, timm, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

models = [
    ("resnet18", 640),
    ("vit_base_patch16_224", 224),
    ("swinv2_tiny_window8_256", 256),
    ("mobilenetv3_small_100", 224),
    ("convnext_tiny", 224),
    ("tf_efficientnetv2_s", 224),
    ("densenet121", 224),
]
val_path = r"E:\Drone_dataset\RFUAV-v2\split_23\val"

print("%-30s %8s %8s" % ("MODEL", "ACC", "SIZE"))
print("-" * 50)
for name, img_size in models:
    ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % name
    if not os.path.exists(ckpt):
        print("%-30s MISSING" % name)
        continue
    model = timm.create_model(name, pretrained=False, num_classes=23)
    model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    model.cuda().eval()
    val_set = datasets.ImageFolder(val_path, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
    ]))
    loader = DataLoader(val_set, batch_size=32, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.cuda())
            correct += (out.argmax(1) == labels.cuda()).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print("%-30s %7.2f%% %6.1fM" % (name, acc, params))
