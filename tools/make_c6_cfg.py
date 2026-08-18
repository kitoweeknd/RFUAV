"""c6 control: NO augmentation, but batch 2 + accum 32 (same regime as c5).
Isolates the micro-batch/BatchNorm effect from the augmentation effect."""
import yaml
d = yaml.safe_load(open(r"D:\ML\RFUAV\configs\train_canonical_v1.yaml", encoding="utf-8"))
d.update(model="resnet50", pretrained=True, num_epochs=50, lr=0.0001,
         batch_size=2, grad_accum_steps=32, image_size=224, amp=True,
         num_workers=2,
         save_path=r"E:\Drone_dataset\RFUAV-v2\runs\c6_resnet50_noaug_b2")
yaml.dump(d, open(r"D:\ML\RFUAV\configs\exp_c6_resnet50_noaug_b2.yaml", "w",
                  encoding="utf-8"), allow_unicode=True)
print("wrote c6 cfg: noaug, batch=2 accum=32, train=%s" % d["train"])
