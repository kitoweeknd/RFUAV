"""YOLOv5 detection backend (vendored inference).

Wraps ``YOLOV5S`` and the vendored YOLOv5 inference utilities. Requires a
``.pt`` checkpoint via ``weight_path``.
"""
import os

import cv2
import numpy as np
import torch

from .base_detector import BaseDetector
from .models import YOLOV5S
from .yolo.basic import (
    Annotator,
    Colors,
    LoadImages,
    Path,
    Profile,
    colorstr,
    letterbox,
    non_max_suppression,
    scale_boxes,
)
from ..logger import colorful_logger


class YoloV5Detector(BaseDetector):
    """Vendored YOLOv5 inference backend."""

    def __init__(self, weight_path, device=None, conf_thres=0.6,
                 imgsz=(640, 640), iou_thres=0.45, max_det=1000,
                 line_thickness=3, hide_labels=True, hide_conf=False):
        super().__init__(device=device, conf_thres=conf_thres)
        if YOLOV5S is None:
            raise ImportError("YOLOv5 backend unavailable (utils.DetModels failed to import)")
        self.logger = colorful_logger('Inference')
        self.model = YOLOV5S(weights=weight_path)
        self.names = self.model.names
        self.imgsz = imgsz
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    @torch.no_grad()
    def detect_batch(self, sources, conf_thres=None):
        """Batched detection over a list of same-shaped BGR frames: one forward
        over the whole batch, then per-image annotation. Falls back to per-frame
        detection when letterboxed frames differ in shape (can't be stacked)."""
        conf = self.conf_thres if conf_thres is None else conf_thres
        if not sources:
            return []
        color = Colors()
        detmodel = self.model
        stride, names = detmodel.stride, detmodel.names
        detmodel.eval()

        letterboxed = [letterbox(src, self.imgsz, stride=stride, auto=True)[0] for src in sources]
        if len({lb.shape for lb in letterboxed}) != 1:
            return [self.yolov5_detect(source=src, conf_thres=conf) for src in sources]

        ims = np.ascontiguousarray(
            np.stack([lb.transpose((2, 0, 1))[::-1] for lb in letterboxed])  # BHWC(BGR)->BCHW(RGB)
        )
        ims = torch.from_numpy(ims).to(detmodel.device).float() / 255

        pred = detmodel(ims)
        pred = non_max_suppression(pred, conf, self.iou_thres, agnostic=False, max_det=self.max_det)

        results = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(sources[i], line_width=self.line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(ims.shape[2:], det[:, :4], sources[i].shape).round()
                for *xyxy, conf_i, cls in reversed(det):
                    c = int(cls)
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf_i:.2f}')
                    annotator.box_label(xyxy, label, color=color(c + 2, True))
            results.append(annotator.result())
        return results

    @torch.no_grad()
    def yolov5_detect(self,
                      source='../example/source/',
                      save_dir='../res',
                      imgsz=(640, 640),
                      conf_thres=0.6,
                      iou_thres=0.45,
                      max_det=1000,
                      line_thickness=3,
                      hide_labels=True,
                      hide_conf=False,
                      ):
        """Single-source detect (backward compatible with the original API):
        a BGR ``np.ndarray`` returns the annotated image; a file/directory path
        writes annotated images to ``save_dir``."""
        color = Colors()
        detmodel = self.model
        stride, names = detmodel.stride, detmodel.names
        # Run inference
        if isinstance(source, np.ndarray):
            detmodel.eval()
            im = letterbox(source, imgsz, stride=stride, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im).to(detmodel.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = detmodel(im)
            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                annotator = Annotator(source, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], source.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                        annotator.box_label(xyxy, label, color=color(c + 2, True))

                # Stream results
                im0 = annotator.result()
                # Save results (image with detections)
            return im0

        else:
            # Ensure the save directory exists
            os.makedirs(save_dir, exist_ok=True)
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            for path, im, im0s, s in dataset:
                im = torch.from_numpy(im).to(detmodel.device)
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                # Inference
                pred = detmodel(im)
                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)
                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = os.path.join(save_dir, p.name)  # im.jpg
                    s += '%gx%g ' % im.shape[2:]  # print string
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                            annotator.box_label(xyxy, label, color=color(c + 2, True))

                    # Stream results
                    im0 = annotator.result()
                    # Save results (image with detections)
                    if save_dir == 'buffer':
                        return im0
                    else:
                        cv2.imwrite(save_path, im0)
                        del im0  # Release memory after saving

            # Print results
            self.logger.log_with_color(f"Results saved to {colorstr('bold', save_dir)}")
