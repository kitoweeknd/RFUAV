import glob
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from graphic.RawDataProcessor import generate_images, frames_to_video
from .build import build_from_cfg, check_cfg
from .logger import colorful_logger
from .metrics.base_metric import EVAMetric
from .trainer import model_init_

try:
    from .DetModels import build_detector
except ImportError:
    build_detector = None


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
            os.makedirs(save_path, exist_ok=True)
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

        name = os.path.splitext(os.path.basename(source))[0]
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

        # Report the classification outcome regardless of save mode: a
        # save=False run previously rendered + inferred everything and then
        # silently discarded it all (no file, no log — the only consumer of
        # the annotated frames is the video writer below).
        majority = max(set(names), key=names.count)
        self.logger.log_with_color(
            f"{name}: {len(names)} frames, majority: {majority} "
            f"({names.count(majority)}/{len(names)}), "
            f"mean confidence: {sum(probs_pct) / len(probs_pct):.2f}%"
        )

        # Annotate + encode only when saving — add_result draws onto the
        # frames in place, and the annotated list has no other consumer.
        # Pipe annotated frames straight to the system ffmpeg (same encoder as
        # save_as_video) instead of imageio.mimsave, which needs the optional
        # imageio-ffmpeg backend that isn't installed. self.save_path is only
        # set when self.save is True (defaults to None), so guard the write.
        if self.save and self.save_path:
            res = [
                self.add_result(res=n, probability=p, image=im)
                for n, p, im in zip(names, probs_pct, images)
            ]
            frames_to_video(res, os.path.join(self.save_path, name + '.mp4'), fps=5)

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

    """Thin facade over the detection backends in ``utils.DetModels``.

    Each backend (yolov5 / fasterrcnn / rtdetr) is a standalone class under
    ``utils/DetModels/``; this class builds the requested one via
    ``build_detector`` and forwards the unified inference API. Kept for backward
    compatibility (``TwoStagesDetector``, older scripts, and README examples).

    Usage::

        det = DetectionModel(model_name='fasterrcnn')      # or 'rtdetr' / 'yolov5'
        annotated = det.detect(source='spectrogram.jpg')   # BGR image with boxes
        frames = det.detect_batch(list_of_bgr_frames)      # batched
    """

    def __init__(self, cfg=None, model_name=None, weight_path=None, conf_thres=0.6):
        self.logger = colorful_logger('Inference')
        cfg = cfg or {}
        if cfg:
            model_name = cfg.get('model_name', model_name)
            weight_path = cfg.get('weight_path', weight_path)
        if build_detector is None:
            raise ImportError("Detection backends unavailable (utils.DetModels failed to import)")

        self.model_name = model_name
        self.weight_path = weight_path
        self.detector = build_detector(model_name, weight_path=weight_path,
                                       cfg=cfg, conf_thres=conf_thres)
        self.device = self.detector.device
        self.names = self.detector.names
        # Backward-compat: expose the underlying model object (was the raw YOLOV5S).
        self.S1model = getattr(self.detector, 'model', None)

    def detect(self, source, conf_thres=None):
        """Single image (path or BGR ndarray) -> annotated BGR image."""
        return self.detector.detect(source, conf_thres=conf_thres)

    def detect_batch(self, sources, conf_thres=None):
        """List of BGR frames -> list of annotated BGR images."""
        return self.detector.detect_batch(sources, conf_thres=conf_thres)

    def yolov5_detect(self, *args, **kwargs):
        """Backward-compatible YOLOv5-only entry point (numpy -> annotated image;
        path/dir -> writes to save_dir). Raises if the active backend isn't yolov5."""
        if not hasattr(self.detector, 'yolov5_detect'):
            raise AttributeError("yolov5_detect is only available for the 'yolov5' backend")
        return self.detector.yolov5_detect(*args, **kwargs)


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

