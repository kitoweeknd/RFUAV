"""Preflight: verify every benchmark model resolves in timm and its
pretrained weights download (cached for the runs)."""
import timm, torch
MODELS = ["mobilenetv3_large_100", "vit_base_patch16_224",
          "swinv2_tiny_window8_256", "convnext_tiny",
          "efficientnetv2_rw_s", "maxvit_tiny_tf_224",
          "eva02_small_patch14_224"]
for m in MODELS:
    try:
        net = timm.create_model(m, pretrained=True, num_classes=46)
        n = sum(p.numel() for p in net.parameters()) / 1e6
        print("OK   %-28s %.1fM params" % (m, n))
        del net
    except Exception as e:
        print("FAIL %-28s %s" % (m, str(e)[:120]))
