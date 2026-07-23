from .models import YOLOV5S
from .base_detector import BaseDetector
from .yolov5_detector import YoloV5Detector
from .fasterrcnn_detector import FasterRCNNDetector
from .rtdetr_detector import RTDetrDetector

SUPPORTED_DETECTORS = ('yolov5', 'fasterrcnn', 'rtdetr')


def build_detector(model_name, weight_path=None, cfg=None, device=None, conf_thres=0.6):
    """Build a detection backend by name.

    - ``yolov5``     -> :class:`YoloV5Detector` (vendored; needs a ``.pt`` weight_path)
    - ``fasterrcnn`` -> :class:`FasterRCNNDetector` (torchvision; COCO-pretrained if no weight)
    - ``rtdetr``     -> :class:`RTDetrDetector` (transformers; HF-pretrained if no weight)

    ``cfg`` (the detector dict from a two-stage JSON) may supply ``num_classes`` /
    ``class_names`` for a custom Faster R-CNN checkpoint.
    """
    cfg = cfg or {}
    name = (model_name or '').lower().strip()
    if name == 'yolov5':
        return YoloV5Detector(weight_path=weight_path, device=device, conf_thres=conf_thres)
    if name == 'fasterrcnn':
        return FasterRCNNDetector(weight_path=weight_path,
                                  num_classes=int(cfg.get('num_classes', 91)),
                                  class_names=cfg.get('class_names'),
                                  device=device, conf_thres=conf_thres)
    if name == 'rtdetr':
        return RTDetrDetector(weight_path=weight_path, device=device, conf_thres=conf_thres)
    raise ValueError(
        f"Unsupported detection model '{model_name}'. Choose from: {SUPPORTED_DETECTORS}.")


__all__ = ['YOLOV5S', 'BaseDetector', 'YoloV5Detector', 'FasterRCNNDetector',
           'RTDetrDetector', 'build_detector', 'SUPPORTED_DETECTORS']
