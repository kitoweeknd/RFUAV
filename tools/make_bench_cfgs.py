"""Generate benchmark configs: one model per family, uniform protocol
(46-class grouped split, pretrained, 50 ep, lr 1e-4, batch 4 + accum 16)."""
import yaml
BASE = r"D:\ML\RFUAV\configs\train_canonical_v1.yaml"
MODELS = [  # (timm name, image_size)
    ("mobilenetv3_large_100", 224),
    ("vit_base_patch16_224", 224),
    ("swinv2_tiny_window8_256", 256),
    ("convnext_tiny", 224),
    ("efficientnetv2_rw_s", 224),
    ("maxvit_tiny_tf_224", 224),
    ("eva02_small_patch14_224", 224),
]
d0 = yaml.safe_load(open(BASE, encoding="utf-8"))
for name, isz in MODELS:
    d = dict(d0)
    d.update(model=name, pretrained=True, num_epochs=50, lr=0.0001,
             batch_size=4, grad_accum_steps=16, image_size=isz, amp=True,
             num_workers=2,
             save_path=r"E:\Drone_dataset\RFUAV-v2\benchmark\runs\bm_" + name)
    p = r"D:\ML\RFUAV\configs\bm_" + name + ".yaml"
    yaml.dump(d, open(p, "w", encoding="utf-8"), allow_unicode=True)
    print("wrote", p)
