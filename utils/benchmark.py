import glob
import os
import time

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from graphic.RawDataProcessor import generate_images
from .build import build_from_cfg, check_cfg
from .logger import colorful_logger
from .metrics.base_metric import EVAMetric
from .trainer import model_init_

try:
    from .DetModels import YOLOV5S
    from .DetModels.yolo.basic import (
        LoadImages,
        Profile,
        Path,
        non_max_suppression,
        Annotator,
        scale_boxes,
        colorstr,
        Colors,
        letterbox,
    )
except ImportError:
    YOLOV5S = None


# Supported image and raw data extensions
image_ext = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
raw_data_ext = ['.iq', '.dat']

class ClassifyModel(nn.Module):
    """
    A class representing a classification model for performing inference and benchmarking using a pre-trained model.

    Attributes:
    - logger (colorful_logger): Logger for logging messages with color.
    - cfg (str): Path to configuration dictionary.
    - device (str): Device to use for inference (CPU or GPU).
    - model (torch.nn.Module): Pre-trained model.
    - save_path (str): Path to save the results.
    - save (bool): Flag to indicate whether to save the results.
    """

    def __init__(self,
                 cfg: str = '../configs/exp1_test.yaml',
                 weight_path: str = '../default.path',
                 save: bool = True,
                 ):

        """
        Initializes the ClassifyModel.

        Parameters:
        - cfg (str): Path to configuration dictionary.
        - weight_path (str): Path to the pre-trained model weights.
        - save (bool): Flag to indicate whether to save the results.
        """

        super().__init__()
        self.logger = self.set_logger

        if check_cfg(cfg):
            self.logger.log_with_color(f"Using config file: {cfg}")
            self.cfg = build_from_cfg(cfg)

        if self.cfg['device'] == 'cuda' and torch.cuda.is_available():
            self.logger.log_with_color("Using GPU for inference")
            self.device = 'cuda'
        else:
            self.logger.log_with_color("Using CPU for inference")
            self.device = "cpu"

        # Normalize class_names into ordered index<->name maps that work for
        # both a list (list order == class index) and a dict (name -> index).
        _class_names = self.cfg['class_names']
        if isinstance(_class_names, dict):
            self._idx_to_name = {int(idx): name for name, idx in _class_names.items()}
            self._name_to_idx = {name: int(idx) for name, idx in _class_names.items()}
        else:
            self._idx_to_name = {i: name for i, name in enumerate(_class_names)}
            self._name_to_idx = {name: i for i, name in enumerate(_class_names)}
        # Ordered class-name tuple in class-index (training) order.
        self._ordered_class_names = tuple(
            self._idx_to_name[i] for i in range(len(self._idx_to_name))
        )

        if os.path.exists(weight_path):
            self.logger.log_with_color(f"Using weight file: {weight_path}")
            self.weight_path = weight_path
        else:
            raise FileNotFoundError(f"weight path: {weight_path} does not exist")

        self.model = self.load_model
        self.model.to(self.device)
        self.model.eval()
        self.save_path = None

        self.save = save

        # Build the eval transform ONCE (was rebuilt per image) and cache fonts,
        # so the per-frame inference loop doesn't repeat this setup work.
        self._transform = transforms.Compose([
            transforms.Resize((self.cfg['image_size'], self.cfg['image_size'])),
            transforms.ToTensor(),
        ])
        self._font_cache = {}
        # Autocast only helps on CUDA; a no-op context on CPU.
        self._use_amp = str(self.device) != 'cpu' and torch.cuda.is_available()

    def inference(self, source='../example/', save_path: str = '../result'):
        """
        Performs inference on the given source data.

        Parameters:
        - source (str): Path to the source data.
        - save_path (str): Path to save the results.
        """
        if self.save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_path = save_path
            self.logger.log_with_color(f"Saving results to: {save_path}")

        if not os.path.exists(source):
            self.logger.warning(f"Source does not exist: {source}")

        # dir detect
        if os.path.isdir(source):
            data_list = glob.glob(os.path.join(source, '*'))

            for data in data_list:
                # detect images in dir
                if is_valid_file(data, image_ext):
                    self.ImgProcessor(data)
                # detect raw datas in dir
                elif is_valid_file(data, raw_data_ext):
                    self.RawdataProcess(data)
                else:
                    continue

        # detect single image
        elif is_valid_file(source, image_ext):
            self.ImgProcessor(source)

        # detect single pack of raw data
        elif is_valid_file(source, raw_data_ext):
            self.RawdataProcess(source)

    def forward(self, img):

        """
        Forward pass through the model.

        Parameters:
        - img (torch.Tensor): Input image tensor.

        Returns:
        - probability (float): Confidence probability of the predicted class.
        - predicted_class_name (str): Name of the predicted class.
        """

        self.model.eval()
        temp = self.model(img)
        probabilities = torch.softmax(temp, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = self._idx_to_name[predicted_class_index]
        probability = probabilities[0][predicted_class_index].item() * 100
        return probability, predicted_class_name

    @property
    def load_model(self):
        """
        Loads the pre-trained model.

        Returns:
        - model (torch.nn.Module): Loaded model.
        """

        self.logger.log_with_color(f"Using device: {self.device}")
        # The checkpoint below is loaded right after, so there is no need to also
        # download ImageNet-pretrained weights (which would fail in offline setups).
        has_weights = os.path.exists(self.weight_path)
        model = model_init_(self.cfg['model'], self.cfg['num_classes'],
                            pretrained=not has_weights, image_size=self.cfg.get('image_size'))

        if has_weights:
            self.logger.log_with_color(f"Loading init weights from: {self.weight_path}")
            state_dict = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            self.logger.success(f"Successfully loaded pretrained weights from: {self.weight_path}")
        else:
            self.logger.warning(f"init weights file not found at: {self.weight_path}. Skipping weight loading.")

        return model

    def ImgProcessor(self, source):
        """
         Performs inference on spectromgram data.

        Parameters:
        - source (str): Path to the image.
        """

        start_time = time.time()

        name = os.path.basename(source)[:-4]
        origin_image = Image.open(source).convert('RGB')
        preprocessed_image = self.preprocess(source)

        with torch.no_grad(), self._autocast():
            temp = self.model(preprocessed_image)

        probabilities = torch.softmax(temp.float(), dim=1)

        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = self._idx_to_name[predicted_class_index]

        end_time = time.time()
        self.logger.log_with_color(f"Inference time: {(end_time - start_time):.8f} sec")
        self.logger.log_with_color(f"{source} contains Drone: {predicted_class_name}, "
                                   f"confidence1: {probabilities[0][predicted_class_index].item() * 100 :.2f} %,"
                                   f" start saving result")

        if self.save:
            res = self.add_result(res=predicted_class_name,
                                  probability=probabilities[0][predicted_class_index].item() * 100,
                                  image=origin_image)

            res.save(os.path.join(self.save_path, name + '.jpg'))

    def RawdataProcess(self, source):
        """
        Transforming raw data into a video and performing inference on video.

        Frames are inferred in batches (was one-at-a-time, which left the GPU
        almost idle) using the cached transform; results are drawn afterwards.

        Parameters:
        - source (str): Path to the raw data.
        """
        images = generate_images(source)
        name = os.path.splitext(os.path.basename(source))[0]
        if not images:
            self.logger.warning(f"No frames generated from: {source}")
            return

        batch_size = int(self.cfg.get('batch_size') or 8)
        names, probs_pct = [], []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                batch = torch.stack(
                    [self._transform(im.convert('RGB')) for im in chunk]
                ).to(self.device, non_blocking=True)
                with self._autocast():
                    outputs = self.model(batch)
                probabilities = torch.softmax(outputs.float(), dim=1)
                conf, idx = probabilities.max(dim=1)
                for j in range(len(chunk)):
                    names.append(self._idx_to_name[int(idx[j].item())])
                    probs_pct.append(float(conf[j].item()) * 100)

        res = [
            self.add_result(res=n, probability=p, image=im)
            for n, p, im in zip(names, probs_pct, images)
        ]

        imageio.mimsave(os.path.join(self.save_path, name + '.mp4'), res, fps=5)

    def add_result(self,
                   res,
                   image,
                   position=(40, 40),
                   font="arial.ttf",
                   font_size=45,
                   text_color=(255, 0, 0),
                   probability=0.0
                   ):
        """
        Adds the inference result to the image.

        Parameters:
        - res (str): Inference result.
        - image (PIL.Image): Input image.
        - position (tuple): Position to add the text.
        - font (str): Font file path.
        - font_size (int): Font size.
        - text_color (tuple): Text color.
        - probability (float): Confidence probability.

        Returns:
        - image (PIL.Image): Image with added result.
        """
        # Load the font once and cache it (was reloaded from disk every frame),
        # falling back to the default font if the TTF isn't installed.
        key = (font, font_size)
        fnt = self._font_cache.get(key)
        if fnt is None:
            try:
                fnt = ImageFont.truetype(font, font_size)
            except Exception:
                fnt = ImageFont.load_default()
            self._font_cache[key] = fnt
        draw = ImageDraw.Draw(image)
        draw.text(position, res + f" {probability:.2f}%", fill=text_color, font=fnt)

        return image

    @property
    def set_logger(self):
        """
        Sets up the logger.

        Returns:
        - logger (colorful_logger): Logger instance.
        """
        logger = colorful_logger('Inference')
        return logger

    def _autocast(self):
        """Mixed-precision context on CUDA; a no-op on CPU."""
        dev = 'cuda' if self._use_amp else 'cpu'
        return torch.autocast(device_type=dev, enabled=self._use_amp)

    def preprocess(self, img):

        image = (img if isinstance(img, Image.Image) else Image.open(img)).convert('RGB')
        preprocessed_image = self._transform(image)

        preprocessed_image = preprocessed_image.to(self.device)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image

    def benchmark(self, data_path, save_path=None):

        """
        Performs benchmarking on the given data and calculates evaluation metrics.

        Parameters:
        - data_path (str): Path to the benchmark data.

        Returns:
        - metrics (dict): Dictionary containing evaluation metrics.
        """
        snrs = os.listdir(data_path)

        if not save_path:
            save_path = os.path.join(data_path, 'benchmark result')

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        num_classes = self.cfg['num_classes']
        cm_raw = np.zeros((num_classes, num_classes), dtype=int)  # accumulated over all snr/CM dirs
        with torch.no_grad():
            for snr in snrs:
                CMS = os.listdir(os.path.join(data_path, snr))
                for CM in CMS:
                    stat_time = time.time()
                    self.model.eval()
                    _dataset = datasets.ImageFolder(
                        root=os.path.join(data_path, snr, CM),
                        transform=self._transform,
                    )
                    dataset = DataLoader(_dataset, batch_size=self.cfg['batch_size'], shuffle=self.cfg['shuffle'])
                    self.logger.log_with_color("Starting Benchmark...")

                    correct = 0
                    total = 0
                    probabilities = []
                    total_labels = []
                    classes_name = self._ordered_class_names
                    # ImageFolder assigns labels 0..k-1 in this folder's own
                    # alphabetical order, which need not match the model's
                    # training class order. Remap folder-local labels to the
                    # global/training indices the predictions live in.
                    remap = torch.full((len(_dataset.classes),), -1, dtype=torch.long)
                    for _name, _local_idx in _dataset.class_to_idx.items():
                        remap[_local_idx] = self._name_to_idx[_name]
                    remap = remap.to(self.device)
                    for images, labels in dataset:
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels = remap[labels]  # folder-local -> global/training indices
                        with self._autocast():
                            outputs = self.model(images).float()
                        for output in outputs:
                            probabilities.append(list(torch.softmax(output, dim=0)))
                        _, predicted = outputs.max(1)
                        for p, t in zip(predicted.cpu(), labels.cpu()):
                            cm_raw[p, t] += 1   # Row = pred, Column = gt
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        total_labels.append(labels)
                    if total == 0:
                        # Empty (or non-image) directory: nothing to score. Skip
                        # it instead of crashing on torch.concat([]) / acc div-by-zero.
                        self.logger.log_with_color(f"Skipping empty directory: {snr}/{CM}")
                        continue
                    _total_labels = torch.concat(total_labels, dim=0)
                    _probabilities = torch.tensor(probabilities)

                    metrics = EVAMetric(preds=_probabilities.to(self.device),
                                        labels=_total_labels,
                                        num_classes=self.cfg['num_classes'],
                                        tasks=('f1', 'precision', 'CM'),
                                        topk=(1, 2, 3),
                                        save_path=save_path,
                                        classes_name=classes_name,
                                        pic_name=f'{snr}_{CM}')
                    metrics['acc'] = 100 * correct / total

                    s = (f'{snr} ' + f'CM: {CM} eva result:' + ' acc: ' + f'{metrics["acc"]}' +
                         ' top-1: ' + f'{metrics["Top-k"]["top1"]}' +
                         ' top-2 ' + f'{metrics["Top-k"]["top2"]}' + ' top-3 ' + f'{metrics["Top-k"]["top3"]}' +
                         ' mAP: ' + f'{metrics["mAP"]["mAP"]}' + ' macro_f1: ' + f'{metrics["f1"]["macro_f1"]}' +
                         ' micro_f1 : ' + f' {metrics["f1"]["micro_f1"]}\n')
                    txt_path = os.path.join(save_path, 'benchmark_result.txt')
                    self.logger.log_with_color(f'cost {(time.time()-stat_time)/60} mins')
                    with open(txt_path, 'a') as file:
                        file.write(s)

                self.logger.log_with_color(f'{CM} Done!')
            self.logger.log_with_color(f'{snr} Done!')

class DetectionModel:

    """
    A common interface for initializing and running different detection models.

    This class provides methods to initialize and run object detection models such as YOLOv5 and Faster R-CNN.
    It allows for easy switching between different models by providing a unified interface.

    Attributes:
    - S1model: The initialized detection model (e.g., YOLOv5S).
    - model_name: The name of the detection model to be used.
    - weight_path: The path to the pre-trained model weights.

    Methods:
    - __init__(self, cfg=None, model_name=None, weight_path=None):
        Initializes the detection model based on the provided configuration or parameters.
        If a configuration dictionary `cfg` is provided, it will be used to set the model name and weight path.
        Otherwise, the `model_name` and `weight_path` parameters can be specified directly.

    - yolov5_detect(self, source='../example/source/', save_dir='../res', imgsz=(640, 640), conf_thres=0.6, iou_thres=0.45, max_det=1000, line_thickness=3, hide_labels=True, hide_conf=False):
        Runs YOLOv5 object detection on the specified source.
        - source: Path to the input image or directory containing images.
        - save_dir: Directory to save the detection results.
        - imgsz: Image size for inference (height, width).
        - conf_thres: Confidence threshold for filtering detections.
        - iou_thres: IoU threshold for non-maximum suppression.
        - max_det: Maximum number of detections per image.
        - line_thickness: Thickness of the bounding box lines.
        - hide_labels: Whether to hide class labels in the output.
        - hide_conf: Whether to hide confidence scores in the output.
    """

    def __init__(self, cfg=None, model_name=None, weight_path=None, conf_thres=0.6):
        self.logger = colorful_logger('Inference')
        cfg = cfg or {}
        if cfg:
            model_name = cfg.get('model_name', model_name)
            weight_path = cfg.get('weight_path', weight_path)

        self.model_name = model_name
        self.weight_path = weight_path
        self.conf_thres = conf_thres
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._use_amp = str(self.device) != 'cpu'
        self.S1model = None   # yolov5 backend
        self.det_model = None  # torchvision / transformers backend
        self.processor = None  # transformers image processor (rtdetr)
        self.names = None      # class-id -> name mapping (dict or list)

        if model_name == 'yolov5':
            if YOLOV5S is None:
                raise ImportError("YOLOv5 backend unavailable (utils.DetModels failed to import)")
            self.S1model = YOLOV5S(weights=weight_path)
            self.S1model.inference = self.yolov5_detect
            self.names = self.S1model.names

        elif model_name == 'fasterrcnn':
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )
            if weight_path and os.path.exists(weight_path):
                num_classes = int(cfg.get('num_classes', 91))
                self.det_model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
                state = torch.load(weight_path, map_location='cpu')
                if isinstance(state, dict) and 'model' in state:
                    state = state['model']
                self.det_model.load_state_dict(state)
                self.names = cfg.get('class_names') or {i: str(i) for i in range(num_classes)}
                self.logger.log_with_color(f"Faster R-CNN: loaded custom weights from {weight_path}")
            else:
                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                self.det_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
                self.names = {i: n for i, n in enumerate(weights.meta['categories'])}
                self.logger.log_with_color("Faster R-CNN: using COCO-pretrained weights")
            self.det_model.to(self.device).eval()

        elif model_name == 'rtdetr':
            try:
                from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            except ImportError as e:
                raise ImportError(
                    "RT-DETR requires the `transformers` package (pip install transformers)."
                ) from e
            ckpt = weight_path or "PekingU/rtdetr_r50vd"
            self.det_model = RTDetrForObjectDetection.from_pretrained(ckpt).to(self.device).eval()
            self.processor = RTDetrImageProcessor.from_pretrained(ckpt)
            self.names = self.det_model.config.id2label
            self.logger.log_with_color(f"RT-DETR: loaded '{ckpt}'")

        else:
            raise ValueError(
                f"Unsupported detection model '{model_name}'. "
                f"Choose from: 'yolov5', 'fasterrcnn', 'rtdetr'."
            )

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

        color = Colors()
        detmodel = self.S1model
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

    @torch.no_grad()
    def yolov5_detect_batch(self,
                            sources,
                            imgsz=(640, 640),
                            conf_thres=0.6,
                            iou_thres=0.45,
                            max_det=1000,
                            line_thickness=3,
                            hide_labels=True,
                            hide_conf=False,
                            ):
        """Batched counterpart of the numpy branch of ``yolov5_detect``.

        Takes a list of BGR frames and runs a single forward over the whole
        batch (instead of one forward per frame), then annotates each image.
        Returns a list of annotated BGR images aligned with ``sources``. Every
        per-image op (letterbox / NMS / scale_boxes / annotate) is identical to
        ``yolov5_detect``; only the model forward is batched. Falls back to
        per-frame detection if the letterboxed frames are not all the same shape
        (so they cannot be stacked)."""
        if not sources:
            return []
        color = Colors()
        detmodel = self.S1model
        stride, names = detmodel.stride, detmodel.names
        detmodel.eval()

        letterboxed = [letterbox(src, imgsz, stride=stride, auto=True)[0] for src in sources]
        if len({lb.shape for lb in letterboxed}) != 1:
            # Different padded sizes can't be stacked; process individually.
            return [self.yolov5_detect(source=src, imgsz=imgsz, conf_thres=conf_thres,
                                       iou_thres=iou_thres, max_det=max_det,
                                       line_thickness=line_thickness,
                                       hide_labels=hide_labels, hide_conf=hide_conf)
                    for src in sources]

        ims = np.ascontiguousarray(
            np.stack([lb.transpose((2, 0, 1))[::-1] for lb in letterboxed])  # BHWC(BGR)->BCHW(RGB)
        )
        ims = torch.from_numpy(ims).to(detmodel.device).float() / 255

        pred = detmodel(ims)
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False, max_det=max_det)

        results = []
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(sources[i], line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(ims.shape[2:], det[:, :4], sources[i].shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=color(c + 2, True))
            results.append(annotator.result())
        return results

    def _autocast(self):
        """Mixed-precision context on CUDA; a no-op on CPU."""
        dev = 'cuda' if self._use_amp else 'cpu'
        return torch.autocast(device_type=dev, enabled=self._use_amp)

    def detect_batch(self, sources, conf_thres=None):
        """Backend-agnostic batched detection.

        `sources`: list of BGR np.ndarray frames. Returns a list of annotated
        BGR np.ndarray images aligned with `sources`. Routes to the active
        backend (yolov5 / fasterrcnn / rtdetr)."""
        conf = self.conf_thres if conf_thres is None else conf_thres
        if self.model_name == 'yolov5':
            return self.yolov5_detect_batch(sources, conf_thres=conf)
        if self.model_name == 'fasterrcnn':
            return self._fasterrcnn_detect_batch(sources, conf)
        if self.model_name == 'rtdetr':
            return self._rtdetr_detect_batch(sources, conf)
        raise ValueError(f"Unsupported detection model '{self.model_name}'")

    def detect(self, source, conf_thres=None):
        """Backend-agnostic single-image detection.

        `source`: image path or BGR np.ndarray. Returns one annotated BGR image."""
        conf = self.conf_thres if conf_thres is None else conf_thres
        img = source if isinstance(source, np.ndarray) else cv2.imread(source)
        if img is None:
            raise ValueError(f"Could not read image: {source}")
        return self.detect_batch([img], conf_thres=conf)[0]

    @torch.no_grad()
    def _fasterrcnn_detect_batch(self, sources, conf_thres):
        tensors = [
            torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                 .permute(2, 0, 1).float().div(255).to(self.device)
            for bgr in sources
        ]
        with self._autocast():
            outputs = self.det_model(tensors)  # torchvision detectors batch a list natively
        results = []
        for bgr, out in zip(sources, outputs):
            scores = out['scores'].detach().cpu().numpy()
            keep = scores >= conf_thres
            boxes = out['boxes'].detach().cpu().numpy()[keep]
            labels = out['labels'].detach().cpu().numpy()[keep]
            results.append(self._draw_boxes(bgr.copy(), boxes, scores[keep], labels))
        return results

    @torch.no_grad()
    def _rtdetr_detect_batch(self, sources, conf_thres):
        pils = [Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) for bgr in sources]
        inputs = self.processor(images=pils, return_tensors="pt").to(self.device)
        with self._autocast():
            outputs = self.det_model(**inputs)
        target_sizes = torch.tensor([[im.height, im.width] for im in pils], device=self.device)
        post = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_thres)
        results = []
        for bgr, res in zip(sources, post):
            results.append(self._draw_boxes(
                bgr.copy(),
                res['boxes'].detach().cpu().numpy(),
                res['scores'].detach().cpu().numpy(),
                res['labels'].detach().cpu().numpy(),
            ))
        return results

    def _draw_boxes(self, bgr, boxes, scores, labels, line_thickness=3):
        """Draw xyxy boxes + labels on a BGR image (fasterrcnn / rtdetr backends)."""
        palette = Colors()
        for box, score, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = (int(v) for v in box[:4])
            lab = int(lab)
            c = palette(lab + 2, True)  # BGR
            cv2.rectangle(bgr, (x1, y1), (x2, y2), c, line_thickness, lineType=cv2.LINE_AA)
            if isinstance(self.names, dict):
                name = self.names.get(lab, str(lab))
            elif self.names is not None and lab < len(self.names):
                name = self.names[lab]
            else:
                name = str(lab)
            cv2.putText(bgr, f"{name} {float(score):.2f}", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, lineType=cv2.LINE_AA)
        return bgr


def is_valid_file(path, total_ext):
    """
    Checks if the file has a valid extension.

    Parameters:
    - path (str): Path to the file.
    - total_ext (list): List of valid extensions.

    Returns:
    - bool: True if the file has a valid extension, False otherwise.
    """
    last_element = os.path.basename(path)
    if any(last_element.lower().endswith(ext) for ext in total_ext):
        return True
    else:
        return False


def get_key_from_value(d, value):
    """
    Gets the key from a dictionary based on the value.

    Parameters:
    - d (dict): Dictionary.
    - value: Value to find the key for.

    Returns:
    - key: Key corresponding to the value, or None if not found.
    """
    for key, val in d.items():
        if val == value:
            return key
    return None

