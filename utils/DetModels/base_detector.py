"""Common base class for detection backends.

Every detector under ``utils.DetModels`` wraps one model and exposes a uniform
inference interface so callers (and ``DetectionModel`` in ``utils.benchmark``)
can switch backends freely:

- ``detect_batch(sources, conf_thres=None) -> list[np.ndarray]`` — annotated BGR
  images, one per input frame.
- ``detect(source, conf_thres=None) -> np.ndarray`` — single annotated BGR image;
  ``source`` may be a BGR ``np.ndarray`` or an image path.

``sources`` are BGR ``np.ndarray`` frames (OpenCV convention).
"""
import cv2
import numpy as np
import torch

from .yolo.basic import Colors


class BaseDetector:
    def __init__(self, device=None, conf_thres=0.6):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self._use_amp = str(self.device) != 'cpu'
        self.names = None  # class-id -> name (dict or list); set by each subclass

    # --- interface -------------------------------------------------------
    def detect_batch(self, sources, conf_thres=None):
        raise NotImplementedError

    def detect(self, source, conf_thres=None):
        """Detect on a single image (BGR ndarray or path) -> annotated BGR image."""
        conf = self.conf_thres if conf_thres is None else conf_thres
        img = source if isinstance(source, np.ndarray) else cv2.imread(source)
        if img is None:
            raise ValueError(f"Could not read image: {source}")
        return self.detect_batch([img], conf_thres=conf)[0]

    # --- shared helpers --------------------------------------------------
    def _autocast(self):
        """Mixed-precision context on CUDA; a no-op on CPU."""
        dev = 'cuda' if self._use_amp else 'cpu'
        return torch.autocast(device_type=dev, enabled=self._use_amp)

    def _class_name(self, label):
        if isinstance(self.names, dict):
            return self.names.get(label, str(label))
        if self.names is not None and label < len(self.names):
            return self.names[label]
        return str(label)

    def draw_boxes(self, bgr, boxes, scores, labels, line_thickness=3):
        """Draw xyxy boxes + labels on a BGR image (cv2). Shared by the
        torchvision / transformers backends; YOLOv5 uses its own Annotator."""
        palette = Colors()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            label = int(label)
            color = palette(label + 2, True)  # BGR
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, line_thickness, lineType=cv2.LINE_AA)
            cv2.putText(bgr, f"{self._class_name(label)} {float(score):.2f}",
                        (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        lineType=cv2.LINE_AA)
        return bgr
