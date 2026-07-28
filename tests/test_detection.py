"""Detection capability smoke tests.

Covers: SUPPORTED_DETECTORS / build_detector dispatch, YOLOv5 structural
forward from yolov5s.yaml (GPU), Faster R-CNN offline end-to-end via a
locally-saved state_dict (no download), the DetectionModel facade, and an
RT-DETR import guard (+ optional networked build).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from utils.DetModels import SUPPORTED_DETECTORS, build_detector

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLO_CFG = REPO_ROOT / "utils" / "DetModels" / "yolo" / "cfg" / "yolov5s.yaml"


def _synthetic_frame(h=384, w=512):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_supported_detectors_and_dispatch():
    assert SUPPORTED_DETECTORS == ("yolov5", "fasterrcnn", "rtdetr")
    with pytest.raises(ValueError):
        build_detector("not_a_detector")


# ---------------------------------------------------------------------------
# YOLOv5 structural forward (no weights needed)
# ---------------------------------------------------------------------------
def test_yolov5_structural_forward(device):
    from utils.DetModels.yolo.model import DetectionModel

    assert YOLO_CFG.exists(), f"missing {YOLO_CFG}"
    model = DetectionModel(cfg=str(YOLO_CFG), ch=3, nc=1).to(device).eval()
    x = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model(x)
    # inference mode returns (pred, ...) or a tensor; predictions carry
    # 5 + nc = 6 channels in the last dim for nc=1.
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert pred.shape[0] == 1
    assert pred.shape[-1] == 6
    assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Faster R-CNN — fully offline (build random-init model, save, load, detect)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offline_frcnn_weights(tmp_path_factory):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    m = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=91)
    p = tmp_path_factory.mktemp("frcnn") / "frcnn_untrained.pth"
    torch.save(m.state_dict(), p)
    return p


def test_fasterrcnn_offline_detect(offline_frcnn_weights):
    det = build_detector("fasterrcnn", weight_path=str(offline_frcnn_weights),
                         cfg={"num_classes": 91}, conf_thres=0.05)
    frame = _synthetic_frame()
    out = det.detect_batch([frame])
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == frame.shape  # annotated frame, same size


def test_detectionmodel_facade_offline(offline_frcnn_weights):
    from utils.benchmark import DetectionModel

    dm = DetectionModel(model_name="fasterrcnn", weight_path=str(offline_frcnn_weights),
                        conf_thres=0.05)
    out = dm.detect_batch([_synthetic_frame()])
    assert isinstance(out, list) and len(out) == 1


# ---------------------------------------------------------------------------
# RT-DETR — import guard always; networked build optional
# ---------------------------------------------------------------------------
def test_rtdetr_importable():
    # The class must import even without transformers installed (lazy import).
    from utils.DetModels.rtdetr_detector import RTDetrDetector  # noqa: F401


@pytest.mark.net
def test_rtdetr_pretrained_build():
    pytest.importorskip("transformers", reason="transformers not installed")
    try:
        det = build_detector("rtdetr", conf_thres=0.5)
    except Exception as e:  # network / HF unreachable
        pytest.skip(f"RT-DETR pretrained build unavailable: {e}")
    out = det.detect_batch([_synthetic_frame()])
    assert isinstance(out, list) and len(out) == 1


# ---------------------------------------------------------------------------
# Faster R-CNN COCO-pretrained end-to-end (networked)
# ---------------------------------------------------------------------------
@pytest.mark.net
def test_fasterrcnn_coco_pretrained():
    try:
        det = build_detector("fasterrcnn", conf_thres=0.5)
    except Exception as e:
        pytest.skip(f"COCO weights unavailable: {e}")
    out = det.detect_batch([_synthetic_frame()])
    assert isinstance(out, list) and len(out) == 1
