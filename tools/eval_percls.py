"""Per-class accuracy of a finished run: eval_percls.py <config.yaml>
Loads runs/<save_path>/best_model.pth, evaluates the valid tree, prints
per-class accuracy sorted ascending."""
import sys, os, yaml, torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
net = timm.create_model(cfg["model"], pretrained=False,
                        num_classes=cfg["num_classes"])
sd = torch.load(os.path.join(cfg["save_path"], "best_model.pth"),
                map_location="cuda", weights_only=True)
net.load_state_dict(sd)
net.cuda().eval()
ds = datasets.ImageFolder(cfg["val"], transform=transforms.Compose([
    transforms.Resize((cfg["image_size"], cfg["image_size"])),
    transforms.ToTensor()]))
dl = DataLoader(ds, batch_size=64, num_workers=2)
hit = {}; tot = {}
with torch.no_grad():
    for x, y in dl:
        p = net(x.cuda()).argmax(1).cpu()
        for yi, pi in zip(y.tolist(), p.tolist()):
            tot[yi] = tot.get(yi, 0) + 1
            hit[yi] = hit.get(yi, 0) + (yi == pi)
pairs = sorted(((hit.get(i, 0) / tot[i], ds.classes[i], tot[i]) for i in tot))
overall = sum(hit.values()) / sum(tot.values())
print("OVERALL %.4f  (%s)" % (overall, os.path.basename(sys.argv[1])))
for acc, name, n in pairs:
    print("  %-30s %6.2f%%  (n=%d)" % (name, acc * 100, n))
