"""Faster R-CNN detection backend (torchvision).

COCO-pretrained by default. Pass ``weight_path`` (and ``num_classes``) to load a
custom checkpoint trained elsewhere. torchvision is already a project dependency,
so this backend needs no extra install.
"""
import os

import cv2
import torch

from .base_detector import BaseDetector
from ..logger import colorful_logger


class FasterRCNNDetector(BaseDetector):
    def __init__(self, weight_path=None, num_classes=91, class_names=None,
                 device=None, conf_thres=0.6):
        super().__init__(device=device, conf_thres=conf_thres)
        self.logger = colorful_logger('Inference')
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        if weight_path and os.path.exists(weight_path):
            num_classes = int(num_classes)
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
            # Loaded straight into load_state_dict below, so the checkpoint is a
            # state_dict (tensors only); weights_only=True is safe and avoids the
            # arbitrary-code-execution risk of unpickling a full object.
            state = torch.load(weight_path, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            self.model.load_state_dict(state)
            self.names = class_names or {i: str(i) for i in range(num_classes)}
            self.logger.log_with_color(f"Faster R-CNN: loaded custom weights from {weight_path}")
        else:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
            self.names = {i: n for i, n in enumerate(weights.meta['categories'])}
            self.logger.log_with_color("Faster R-CNN: using COCO-pretrained weights")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect_batch(self, sources, conf_thres=None):
        conf = self.conf_thres if conf_thres is None else conf_thres
        if not sources:
            return []
        tensors = [
            torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float().div(255).to(self.device)
            for bgr in sources
        ]
        with self._autocast():
            outputs = self.model(tensors)  # torchvision detectors batch a list natively
        results = []
        for bgr, out in zip(sources, outputs):
            scores = out['scores'].detach().cpu().numpy()
            keep = scores >= conf
            boxes = out['boxes'].detach().cpu().numpy()[keep]
            labels = out['labels'].detach().cpu().numpy()[keep]
            results.append(self.draw_boxes(bgr.copy(), boxes, scores[keep], labels))
        return results
