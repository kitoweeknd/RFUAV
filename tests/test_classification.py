"""Classification capability smoke tests.

Covers: model_init_ build+forward across timm backbones, is_supported_model,
ClassifyModel forward + inference (from-scratch .pth), benchmark tree, the
EVAMetric facade, and a 1-epoch CustomTrainer run (GPU/slow).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.build import is_supported_model
from utils.trainer import model_init_


# ---------------------------------------------------------------------------
# model_init_ across a representative spread of timm backbones
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,size",
    [
        ("resnet18", 64),
        ("resnet50", 64),
        ("mobilenetv3_small_100", 64),
        ("vit_base_patch16_224", 224),
        ("swinv2_tiny_window8_256", 256),
    ],
)
def test_model_init_forward(name, size, device):
    model = model_init_(name, num_class=5, pretrained=False, image_size=size).to(device).eval()
    x = torch.randn(2, 3, size, size, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_model_init_backward(device):
    """One optimizer step must reduce a trivial loss (grad path works)."""
    model = model_init_("resnet18", num_class=5, pretrained=False, image_size=64).to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 3, 64, 64, device=device)
    target = torch.randint(0, 5, (4,), device=device)
    crit = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(3):
        opt.zero_grad()
        loss = crit(model(x), target)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_is_supported_model():
    assert is_supported_model("resnet18")
    assert is_supported_model("vit_base_patch16_224")
    assert not is_supported_model("definitely_not_a_real_model_xyz")
    assert not is_supported_model(None)  # non-str guarded


# ---------------------------------------------------------------------------
# ClassifyModel: forward + inference using the from-scratch .pth
# ---------------------------------------------------------------------------
def test_classifymodel_forward(tiny_classifier, device):
    from utils.benchmark import ClassifyModel

    clf = ClassifyModel(cfg=str(tiny_classifier["cfg"]),
                        weight_path=str(tiny_classifier["weights"]),
                        save=False)
    size = tiny_classifier["image_size"]
    img = torch.randn(1, 3, size, size, device=clf.device)
    prob, name = clf.forward(img)
    assert 0.0 <= prob <= 100.0
    assert name in tiny_classifier["class_names"]


def test_classifymodel_inference_dir(tiny_classifier, tmp_path):
    from utils.benchmark import ClassifyModel

    clf = ClassifyModel(cfg=str(tiny_classifier["cfg"]),
                        weight_path=str(tiny_classifier["weights"]),
                        save=True)
    # inference over the first class dir (a handful of PNGs) → annotated .jpg out
    src = sorted(p for p in tiny_classifier["data_root"].iterdir() if p.is_dir())[0]
    out = tmp_path / "clf_out"
    clf.inference(source=str(src), save_path=str(out))
    jpgs = list(out.glob("*.jpg"))
    assert jpgs, "inference produced no annotated images"


@pytest.mark.slow
def test_classifymodel_benchmark(tiny_classifier, benchmark_tree, tmp_path):
    """benchmark() walks <snr>/<CM>/<ImageFolder> and computes topk=(1,2,3)."""
    from utils.benchmark import ClassifyModel

    clf = ClassifyModel(cfg=str(tiny_classifier["cfg"]),
                        weight_path=str(tiny_classifier["weights"]),
                        save=False)
    out = tmp_path / "bench_out"
    out.mkdir()
    # Should complete without raising; return value/aggregation is impl detail.
    clf.benchmark(data_path=str(benchmark_tree), save_path=str(out))


# ---------------------------------------------------------------------------
# metrics facade
# ---------------------------------------------------------------------------
def test_eva_metric_facade(tmp_path):
    from utils.metrics.base_metric import EVAMetric

    n, nc = 100, 5
    rng = np.random.default_rng(0)
    preds = torch.from_numpy(rng.standard_normal((n, nc)).astype(np.float32))
    labels = torch.from_numpy(rng.integers(0, nc, size=n))
    res = EVAMetric(
        preds=preds,
        labels=labels,
        tasks=("f1", "precision", "CM"),
        num_classes=nc,
        topk=(1, 3, 5),
        save_path=str(tmp_path) + "/",
        classes_name=("A", "B", "C", "D", "E"),
        pic_name="test",
    )
    assert isinstance(res, dict)
    assert res, "EVAMetric returned an empty result dict"


# ---------------------------------------------------------------------------
# end-to-end 1-epoch training (GPU, slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_custom_trainer_one_epoch(tiny_classifier):
    from utils.trainer import CustomTrainer

    trainer = CustomTrainer(cfg=str(tiny_classifier["cfg"]))
    trainer.train()  # num_epochs=1 from cfg; must complete without raising
