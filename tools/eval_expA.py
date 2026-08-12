import torch, timm, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

models = {
    "resnet50": (r"E:\Drone_dataset\RFUAV-v2\checkpoints\expA_resnet50\best_model.pth", 640),
    "vit_base_patch16_224": (r"E:\Drone_dataset\RFUAV-v2\checkpoints\expA_vit\best_model.pth", 224),
    "swinv2_tiny_window8_256": (r"E:\Drone_dataset\RFUAV-v2\checkpoints\expA_swin\best_model.pth", 256),
}
val_path = r"E:\Drone_dataset\RFUAV-v2\split_dji5_flat\val"

for name, (ckpt, img_size) in models.items():
    if not os.path.exists(ckpt):
        print(name, "MISSING checkpoint")
        continue
    model = timm.create_model(name, pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(ckpt, map_location="cuda"), strict=False)
    model.cuda().eval()
    val_set = datasets.ImageFolder(val_path, transform=transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
    ]))
    loader = DataLoader(val_set, batch_size=8, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = model(imgs.cuda())
            correct += (out.argmax(1) == labels.cuda()).sum().item()
            total += labels.size(0)
    print("%s: %d/%d = %.2f%%" % (name, correct, total, 100*correct/total))
