"""Validation config: same recipe as c8a (batch4+accum16) but on thumbnails.
Equivalence gate: best top-1 within the c8a replicate band (61.45-62.97 +/-1.5)."""
import yaml
d = yaml.safe_load(open(r"D:\ML\RFUAV\configs\train_canonical_v1.yaml", encoding="utf-8"))
d.update(model="resnet50", pretrained=True, num_epochs=50, lr=0.0001,
         batch_size=4, grad_accum_steps=16, image_size=224, amp=True, num_workers=4,
         train=r"E:\Drone_dataset\RFUAV-v2\benchmark\train_dataset_v1_t256\train",
         val=r"E:\Drone_dataset\RFUAV-v2\benchmark\train_dataset_v1_t256\valid",
         save_path=r"E:\Drone_dataset\RFUAV-v2\benchmark\runs\vt_resnet50_t256")
yaml.dump(d, open(r"D:\ML\RFUAV\configs\vt_resnet50_t256.yaml", "w", encoding="utf-8"),
          allow_unicode=True)
print("wrote vt_resnet50_t256.yaml (classes=%d)" % d["num_classes"])
