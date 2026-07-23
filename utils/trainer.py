"""
The base trainer class `Basetrainer` and a custom trainer class `CustomTrainer` for training and validating image classification models.
"""
from copy import deepcopy
from datetime import datetime
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml

from .build import build_from_cfg, check_cfg
from .logger import colorful_logger
from .metrics.base_metric import EVAMetric
from .DetModels.yolo.torch_utils import ModelEMA


TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def _make_grad_scaler(enabled: bool):
    """Create a CUDA GradScaler compatibly across torch versions.

    ``torch.amp.GradScaler('cuda', ...)`` is the modern API (torch >= 2.4);
    older versions (torch 2.0-2.3) only expose ``torch.cuda.amp.GradScaler``.
    Prefer the new form and fall back so the stated ``torch >= 2.0.1``
    requirement keeps working without emitting the 2.4+ deprecation warning.
    """
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Basetrainer:

    """
    Base trainer class for initializing the model, dataset, optimizer, and performing training and validation.

    Parameters:
    - model (str): any timm model name (see ``timm.list_models()``), e.g.
                  "resnet50", "vit_base_patch16_224", "convnext_tiny",
                  "swinv2_tiny_window8_256", "efficientnetv2_rw_s"
    - train_path (str): Path to the training dataset
    - val_path (str): Path to the validation dataset
    - num_class (int): Number of classes
    - save_path (str): Path to save the model
    - weight_path (str, optional): Path to pre-trained weights, default is None
    - log_file (str, optional): Path to the log file, default is "train.log"
    - device (str, optional): Device to use, "cuda" or "cpu", default is "cuda"
    - criterion (torch.nn.Module, optional): Loss function, default is `nn.CrossEntropyLoss()`
    - pretrained (bool, optional): Whether to use pre-trained model, default is `True`
    - batch_size (int, optional): Batch size, default is 8
    - shuffle (bool, optional): Whether to shuffle the data, default is `False`
    - image_size (int, optional): Image size, default is 224
    - lr (float, optional): Learning rate, default is 0.0001
    """

    def __init__(self,
                 model: str,
                 train_path: str,
                 val_path: str,
                 num_class: int,
                 save_path: str,
                 weight_path: str = "",
                 log_file: str = "train.log",
                 device: str = "cuda",
                 criterion=nn.CrossEntropyLoss(),
                 pretrained: bool = True,
                 batch_size: int = 8,
                 shuffle: bool = False,
                 image_size: int = 224,
                 lr: float = 0.0001,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 amp: bool = False,
                 val_interval: int = 1,
                 metrics_interval: int = 1,
                 cudnn_benchmark: bool = True,
                 compile_model: bool = False,
                 grad_accum_steps: int = 1,
                 grad_clip_norm: float = 0.0,
                 ema_decay: float = 0.0,
                 prefetch_factor: int = 2
                 ):

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.amp = amp
        self.val_interval = max(1, int(val_interval))
        self.metrics_interval = max(1, int(metrics_interval))
        self.cudnn_benchmark = cudnn_benchmark
        self.compile_model = compile_model
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.grad_clip_norm = float(grad_clip_norm)
        self.ema_decay = float(ema_decay)
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.num_class = num_class
        self.save_path = save_path
        self.best_acc = 0
        self.best_loss = 1e6
        self.best_epoch = 0
        self.best_model = None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.logger = self.set_logger(os.path.join(save_path, log_file))
        self.criterion = criterion  # initializing the loss function
        self.set_up(model=model, train_path=train_path, val_path=val_path,
                    pretrained=pretrained, weight_path=weight_path)

    def set_up(self, train_path, val_path, pretrained, weight_path, model='resnet18'):

        """
        Initialize the model, dataset, and optimizer.

        Parameters:
        - train_path (str): Path to the training dataset
        - val_path (str): Path to the validation dataset
        - pretrained (bool): Whether to use pre-trained model
        - weight_path (str): Path to pre-trained weights
        - model (str): Model name, default is "resnet18"
        """

        self.logger.log_with_color(f"Loading model: {model}")

        if weight_path and os.path.exists(weight_path):
            pretrained = False

        if not isinstance(pretrained, bool):
            self.logger.warning("Invalid pretrained flag, fallback to True")
            pretrained = True

        torch.backends.cudnn.benchmark = self.cudnn_benchmark

        self.model = model_init_(model_name=model, num_class=self.num_class,
                                 pretrained=pretrained, image_size=self.image_size)

        if weight_path and os.path.exists(weight_path):
            self.load_pretrained_weights(weight_path)
            self.logger.log_with_color(f"Loading pretrained weights from: {weight_path}")

        self.model.to(self.device)
        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        self.logger.log_with_color(f"{model} loaded onto device: {self.device}")

        # initializing the dataset
        self.logger.log_with_color(f"Loading dataset from: {train_path} and {val_path}")
        _train_set = datasets.ImageFolder(root=train_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))

        self.train_set = DataLoader(
            _train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

        _val_set = datasets.ImageFolder(root=val_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))
        self.val_set = DataLoader(
            _val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

        # initializing optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.grad_scaler = _make_grad_scaler(enabled=self.amp)
        self.ema = None
        if self.ema_decay > 0:
            self.ema = ModelEMA(self.model, decay=self.ema_decay)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.logger.log_with_color(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            self.optimizer.zero_grad(set_to_none=True)
            for batch_idx, (images, labels) in enumerate(self.train_set):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # forward
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    outputs = self.model(images)
                    raw_loss = self.criterion(outputs, labels)
                    loss = raw_loss / self.grad_accum_steps

                # backward
                if self.amp:
                    self.grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.amp:
                        self.grad_scaler.unscale_(self.optimizer)
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    if self.amp:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)

                # acc & loss
                running_loss += raw_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % self.grad_accum_steps != 0:
                if self.amp:
                    self.grad_scaler.unscale_(self.optimizer)
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.amp:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.log_with_color(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            if (epoch + 1) % self.val_interval == 0:
                metrics = self.val
                self.logger.log_with_color(f' Validation Loss: {metrics["total_loss"]:.4f},')
                self.logger.log_with_color(f' Validation Accuracy: {metrics["acc"]:.2f}%,')
                if (epoch + 1) % self.metrics_interval == 0:
                    self.logger.log_with_color(f' Validation macro_F1: {metrics["f1"]["macro_f1"]}')
                    self.logger.log_with_color(f' Validation micro_F1: {metrics["f1"]["micro_f1"]}')
                    self.logger.log_with_color(f' Validation mAP: {metrics["mAP"]["mAP"]}')
                    self.logger.log_with_color(f' Validation Top-k Accuracy: {metrics["Top-k"]}')
                else:
                    metrics = {"acc": metrics["acc"], "total_loss": metrics["total_loss"]}
                self.save_model(metrics, epoch)

    @property
    def val(self):
        self.logger.log_with_color("Starting validation...")
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probabilities = []
        val_total_labels = []
        with torch.no_grad():
            for val_images, val_labels in self.val_set:
                val_images = val_images.to(self.device, non_blocking=True)
                val_labels = val_labels.to(self.device, non_blocking=True)
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    val_outputs = eval_model(val_images)
                    val_loss += self.criterion(val_outputs, val_labels).item()
                val_probabilities.append(torch.softmax(val_outputs, dim=1))
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
                val_total_labels.append(val_labels)
        _val_total_labels = torch.cat(val_total_labels, dim=0)
        _val_probabilities = torch.cat(val_probabilities, dim=0)
        tasks = ('f1', 'precision', 'CM') if self.metrics_interval == 1 else ('f1', 'precision')
        metrics = EVAMetric(preds=_val_probabilities.to(self.device),
                            labels=_val_total_labels,
                            num_classes=self.num_class,
                            tasks=tasks,
                            topk=(1, 3, 5),
                            save_path=self.save_path,
                            classes_name=self.train_set.dataset.classes)

        metrics['acc'] = 100 * val_correct / val_total
        metrics['total_loss'] = val_loss / len(self.val_set)
        return metrics

    def save_model(self, metrics, epoch):

        """
        Save the model after each epoch and track the best model based on validation accuracy.
        """

        # Save the same weights `val` scores: the EMA shadow if EMA is active,
        # else the live model. Unwrap torch.compile so the state_dict has plain
        # keys (no `_orig_mod.` prefix) and reloads into a fresh model.
        save_target = self.ema.ema if getattr(self, 'ema', None) is not None else self.model
        save_target = getattr(save_target, '_orig_mod', save_target)
        state_dict = save_target.state_dict()

        checkpoint_path = os.path.join(self.save_path, f'{save_target._get_name()}_epoch_{epoch + 1}.pth')
        self.logger.log_with_color(
            f'Model saved at {checkpoint_path} (Validation Accuracy: {metrics["acc"]:.2f}%)')
        torch.save(state_dict, checkpoint_path)

        # Save the best model if current validation accuracy is higher than the best recorded one
        if metrics["acc"] > self.best_acc:
            self.best_acc = metrics["acc"]
            self.best_model = state_dict
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.best_model, best_model_path)
            self.logger.log_with_color(f'New best model saved with Accuracy: {metrics["acc"]:.2f}%')

    def set_logger(self, log_file):

        """
        Set up the logger.

        Parameters:
        - log_file (str): Path to the log file

        Returns:
        - logger (colorful_logger): Logger object
        """

        logger = colorful_logger(name='Train', logfile=log_file)
        return logger

    def load_pretrained_weights(self, weight_path: str):

        if os.path.exists(weight_path):
            self.logger.log_with_color(f"Loading pretrained weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.logger.success(f"Successfully loaded pretrained weights from: {weight_path}")
        else:
            self.logger.warning(f"Pretrained weights file not found at: {weight_path}. Skipping weight loading.")


def model_init_(model_name, num_class, pretrained=True, image_size=None):
    """
    Build a classification model via ``timm`` (1000+ architectures).

    ``model_name`` is any name from ``timm.list_models()`` (e.g. ``resnet50``,
    ``vit_base_patch16_224``, ``convnext_tiny``, ``swinv2_tiny_window8_256``).
    timm swaps the classifier head to ``num_class`` automatically.

    Parameters:
    - model_name (str): timm model name
    - num_class (int): number of output classes
    - pretrained (bool, optional): load pretrained backbone weights, default True
    - image_size (int, optional): input size forwarded as ``img_size`` to models
      that support a configurable input (ViT/Swin/...); ignored by fixed CNNs.

    Returns:
    - model (torch.nn.Module)
    """
    name = str(model_name).strip()
    kwargs = dict(pretrained=pretrained, num_classes=num_class)
    if image_size:
        try:
            return timm.create_model(name, img_size=int(image_size), **kwargs)
        except TypeError:
            pass  # this architecture has no configurable img_size (e.g. CNNs)
    return timm.create_model(name, **kwargs)


class CustomTrainer(Basetrainer):

    """
    Custom trainer class that extends the `Basetrainer` class. It initializes the trainer with configuration parameters
    and provides additional functionality.

    Parameters:
    - cfg (str): Configuration file path
    """

    def __init__(self,
                 cfg: str,
                 ):
        if check_cfg(cfg):
            self.parameters = build_from_cfg(cfg)
            super().__init__(
                model=self.parameters['model'],
                train_path=self.parameters['train'],
                val_path=self.parameters['val'],
                num_class=self.parameters['num_classes'],
                save_path=self.parameters['save_path'],
                weight_path=self.parameters['weights'],
                pretrained=self.parameters.get('pretrained', True),
                device=self.parameters['device'],
                batch_size=self.parameters['batch_size'],
                shuffle=self.parameters['shuffle'],
                image_size=self.parameters['image_size'],
                lr=self.parameters['lr'],
                num_workers=self.parameters.get('num_workers', 4),
                pin_memory=self.parameters.get('pin_memory', True),
                amp=self.parameters.get('amp', False),
                val_interval=self.parameters.get('val_interval', 1),
                metrics_interval=self.parameters.get('metrics_interval', 1),
                cudnn_benchmark=self.parameters.get('cudnn_benchmark', True),
                compile_model=self.parameters.get('compile_model', False),
                grad_accum_steps=self.parameters.get('grad_accum_steps', 1),
                grad_clip_norm=self.parameters.get('grad_clip_norm', 0.0),
                ema_decay=self.parameters.get('ema_decay', 0.0),
                prefetch_factor=self.parameters.get('prefetch_factor', 2),
            )
        else:
            raise ValueError(f"Invalid configuration file: {cfg}")

        self.class_idx = self.train_set.dataset.class_to_idx
        if self.save_yaml:
            self.logger.log_with_color(f"Saving yaml file at {self.parameters['save_path']}")

    @property
    def save_yaml(self):

        self.parameters['class_names'] = self.class_idx
        with open(os.path.join(self.save_path, 'config.yaml'), 'w', encoding='utf-8') as file:
            yaml.dump(self.parameters, file, allow_unicode=True)
        return True

    def train(self):
        # Delegate to Basetrainer.train, which does the correct per-step
        # zero_grad, gradient accumulation/clipping and EMA update. (The old
        # inline loop here skipped per-step zero_grad — gradients accumulated
        # across the whole epoch — and never updated the EMA used by `val`.)
        super().train(self.parameters['num_epochs'])
