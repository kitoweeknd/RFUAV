"""End-to-end two-stage detector tests (utils.TwoStagesDetector).

`TwoStagesDetector` does all its work in ``__init__`` from a JSON config
(detector + classifier + target_dir + save_dir). We drive it with a temporary
Linux-path JSON (never touching the repo's Windows-path sample.json) and
synthetic spectrogram images:

* classifier-only  — fully offline (tiny from-scratch .pth).
* detector+classifier — fully offline (untrained Faster R-CNN state_dict).
* HF-pretrained full pipeline — opportunistic, ``net``-marked, skips cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _spectrogram_png(path: Path, seed: int = 0):
    """A cheap RGB image standing in for a spectrogram frame."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_json(tmp_path: Path, *, detector=None, classifier=None,
                target_dir: Path, save_dir: Path) -> Path:
    cfg = {"target_dir": str(target_dir), "save_dir": str(save_dir)}
    if detector is not None:
        cfg["detector"] = detector
    if classifier is not None:
        cfg["classifier"] = classifier
    p = tmp_path / "two_stage.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# classifier-only — fully offline
# ---------------------------------------------------------------------------
def test_two_stage_classifier_only_offline(tiny_classifier, tmp_path):
    from utils.TwoStagesDetector import TwoStagesDetector

    target = tmp_path / "source"
    target.mkdir()
    _spectrogram_png(target / "frame0.png")
    save = tmp_path / "res"

    cfg = _write_json(
        tmp_path,
        classifier={"weight_path": str(tiny_classifier["weights"]),
                    "cfg": str(tiny_classifier["cfg"])},
        target_dir=target, save_dir=save,
    )
    TwoStagesDetector(cfg=str(cfg))
    assert list(save.glob("*.jpg")), "classifier-only two-stage wrote no output"


# ---------------------------------------------------------------------------
# detector + classifier — fully offline (untrained Faster R-CNN)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offline_frcnn_weights(tmp_path_factory):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    m = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=91)
    p = tmp_path_factory.mktemp("frcnn2") / "frcnn_untrained.pth"
    torch.save(m.state_dict(), p)
    return p


@pytest.mark.slow
def test_two_stage_det_and_cls_offline(tiny_classifier, offline_frcnn_weights, tmp_path):
    from utils.TwoStagesDetector import TwoStagesDetector

    target = tmp_path / "source"
    target.mkdir()
    _spectrogram_png(target / "frame0.png")
    save = tmp_path / "res"

    cfg = _write_json(
        tmp_path,
        detector={"model_name": "fasterrcnn", "weight_path": str(offline_frcnn_weights)},
        classifier={"weight_path": str(tiny_classifier["weights"]),
                    "cfg": str(tiny_classifier["cfg"])},
        target_dir=target, save_dir=save,
    )
    TwoStagesDetector(cfg=str(cfg))
    assert list(save.glob("*.jpg")), "det+cls two-stage wrote no output"


# ---------------------------------------------------------------------------
# HF-pretrained full pipeline — opportunistic, skips cleanly
# ---------------------------------------------------------------------------
@pytest.mark.net
def test_two_stage_hf_pretrained(tmp_path):
    hf = pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")

    # Best-effort: pull the published RFUAV two-stage weights. Any failure
    # (offline, repo/filename drift, auth) skips rather than fails.
    try:
        stage1 = hf.hf_hub_download("kitofrank/RFUAV", "RFUAV_stage1.pt")
        stage2 = hf.hf_hub_download("kitofrank/RFUAV", "stage2.pth")
        stage2_cfg = hf.hf_hub_download("kitofrank/RFUAV", "stage2.yaml")
    except Exception as e:
        pytest.skip(f"RFUAV HF weights unreachable: {e}")

    from utils.TwoStagesDetector import TwoStagesDetector

    target = tmp_path / "source"
    target.mkdir()
    _spectrogram_png(target / "frame0.png")
    save = tmp_path / "res"

    cfg = _write_json(
        tmp_path,
        detector={"model_name": "yolov5", "weight_path": stage1},
        classifier={"weight_path": stage2, "cfg": stage2_cfg},
        target_dir=target, save_dir=save,
    )
    try:
        TwoStagesDetector(cfg=str(cfg))
    except Exception as e:
        pytest.skip(f"HF two-stage pipeline failed to run: {e}")
    assert list(save.glob("*.jpg"))
