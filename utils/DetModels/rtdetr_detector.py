"""RT-DETR detection backend (HuggingFace transformers).

``weight_path`` may be a local directory or a HuggingFace hub id; it defaults to
``"PekingU/rtdetr_r50vd"`` (downloaded on first use). ``transformers`` is imported
lazily so this backend's absence never breaks the others.
"""
import cv2
import torch
from PIL import Image

from .base_detector import BaseDetector
from ..logger import colorful_logger


class RTDetrDetector(BaseDetector):
    def __init__(self, weight_path=None, device=None, conf_thres=0.6):
        super().__init__(device=device, conf_thres=conf_thres)
        self.logger = colorful_logger('Inference')
        try:
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as e:
            raise ImportError(
                "RT-DETR requires the `transformers` package (pip install transformers)."
            ) from e
        ckpt = weight_path or "PekingU/rtdetr_r50vd"
        self.model = RTDetrForObjectDetection.from_pretrained(ckpt).to(self.device).eval()
        self.processor = RTDetrImageProcessor.from_pretrained(ckpt)
        self.names = self.model.config.id2label
        self.logger.log_with_color(f"RT-DETR: loaded '{ckpt}'")

    @torch.no_grad()
    def detect_batch(self, sources, conf_thres=None):
        conf = self.conf_thres if conf_thres is None else conf_thres
        if not sources:
            return []
        pils = [Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) for bgr in sources]
        inputs = self.processor(images=pils, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[im.height, im.width] for im in pils], device=self.device)
        post = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf)
        results = []
        for bgr, res in zip(sources, post):
            results.append(self.draw_boxes(
                bgr.copy(),
                res['boxes'].detach().cpu().numpy(),
                res['scores'].detach().cpu().numpy(),
                res['labels'].detach().cpu().numpy(),
            ))
        return results
