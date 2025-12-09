"""
The base trainer class `Basetrainer` and a custom trainer class `CustomTrainer` for training and validating image classification models.
"""
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import os
import yaml
from utils.build import build_from_cfg, check_cfg
from utils.logger import colorful_logger
import cv2
from abc import abstractmethod
from .metrics.base_metric import EVAMetric
import sys

from tqdm import tqdm
from pathlib import Path
from utils.DetModels.yolo import DetectionModel
import numpy as np
from torch.optim import lr_scheduler
import random
import math
import time
from datetime import datetime
from copy import deepcopy
from utils.DetModels.yolo.general import (yaml_save, init_seeds, check_dataset,check_suffix,
                                          check_img_size, labels_to_class_weights, labels_to_image_weights)
from utils.DetModels.yolo.basic import colorstr, yolo_init
from utils.DetModels.yolo.torch_utils import (torch_distributed_zero_first,smart_optimizer,
                                              de_parallel, EarlyStopping, ModelEMA, select_device)
from utils.DetModels.yolo.dataloader import create_dataloader
from utils.DetModels.yolo.autoanchor import check_anchors
from utils.DetModels.yolo.loss import ComputeLoss
import utils.DetModels.yolo.val as validate
from utils.DetModels.yolo.metrics import fitness
from utils.DetModels.yolo.callbacks import Callbacks


current_dir = os.path.dirname(os.path.abspath(__file__))
METRIC = os.path.join(current_dir, './metrics')
sys.path.append(METRIC)
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


class Basetrainer:

    """
    Base trainer class for initializing the model, dataset, optimizer, and performing training and validation.

    Parameters:
    - model (str): Model name, supported models include "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                  "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
                  "swin_v2_t", "swin_v2_s", "swin_v2_b", "mobilenet_v3_small", "mobilenet_v3_large"
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
                 lr: float = 0.0001
                 ):

        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
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

        if os.path.exists(weight_path):
            pretrained = False

        if not os.path.exists(pretrained):
            self.logger.log_with_color("Pretrained model not found, using default weight")
            pretrained = True

        self.model = model_init_(model_name=model, num_class=self.num_class, pretrained=pretrained)

        if os.path.exists(weight_path):
            self.load_pretrained_weights(weight_path)
            self.logger.log_with_color(f"Loading pretrained weights from: {weight_path}")

        self.model.to(self.device)
        self.logger.log_with_color(f"{model} loaded onto device: {self.device}")

        # initializing the dataset
        self.logger.log_with_color(f"Loading dataset from: {train_path} and {val_path}")
        _train_set = datasets.ImageFolder(root=train_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))

        self.train_set = DataLoader(_train_set, batch_size=self.batch_size, shuffle=self.shuffle)

        _val_set = datasets.ImageFolder(root=val_path, transform=transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]))
        self.val_set = DataLoader(_val_set, batch_size=self.batch_size, shuffle=self.shuffle)

        # initializing optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @abstractmethod
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.logger.log_with_color(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_set:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # backward
                loss.backward()
                self.optimizer.step()

                # acc & loss
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.log_with_color(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            metrics = self.val
            self.logger.log_with_color(f'Validation Loss: {metrics["loss"]:.4f}, Validation Accuracy: {metrics["acc"]:.2f}%')
            self.save_model(metrics['acc'], epoch)

    @property
    def val(self):
        self.logger.log_with_color("Starting validation...")
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probabilities = []
        val_total_labels = []
        with torch.no_grad():
            for val_images, val_labels in self.val_set:
                val_images, val_labels = val_images.to(self.device), val_labels.to(self.device)
                val_outputs = self.model(val_images)
                for val_output in val_outputs:
                    val_probabilities.append(list(torch.softmax(val_output, dim=0)))
                val_loss += self.criterion(val_outputs, val_labels).item()
                _, val_predicted = val_outputs.max(1)
                val_total += val_labels.size(0)
                val_correct += val_predicted.eq(val_labels).sum().item()
                val_total_labels.append(val_labels)
        _val_total_labels = torch.concat(val_total_labels, dim=0)
        _val_probabilities = torch.tensor(val_probabilities)
        metrics = EVAMetric(preds=_val_probabilities.to(self.device),
                            labels=_val_total_labels,
                            num_classes=self.num_class,
                            tasks=('f1', 'precision'),
                            topk=(1, 3, 5),
                            save_path=self.save_path,
                            classes_name=self.train_set.dataset.classes)

        metrics['acc'] = 100 * val_correct / val_total
        metrics['total_loss'] = val_loss / len(self.val_set)
        return metrics

    def save_model(self, val_acc, epoch):

        """
        Save the model after each epoch and track the best model based on validation accuracy.
        """

        checkpoint_path = os.path.join(self.save_path, f'{self.model._get_name()}_epoch_{epoch + 1}.pth')
        self.logger.log_with_color(f'Model saved at {checkpoint_path} (Validation Accuracy: {val_acc["acc"]:.2f}%)')
        torch.save(self.model.state_dict(), checkpoint_path)

        # Save the best model if current validation accuracy is higher than the best recorded one
        if val_acc["acc"] > self.best_acc:
            self.best_acc = val_acc["acc"]
            self.best_model = self.model.state_dict()
            best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.best_model, best_model_path)
            self.logger.log_with_color(f'New best model saved with Accuracy: {val_acc["acc"]:.2f}%')

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
            state_dict = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.log_with_color(f"Successfully loaded pretrained weights from: {weight_path}")
        else:
            self.logger.log_with_color(f"Pretrained weights file not found at: {weight_path}. Skipping weight loading.")


def model_init_(model_name, num_class, pretrained=True):
    """
    Initialize the model.

    Parameters:
    - model_name (str): Model name
    - num_class (int): Number of classes
    - pretrained (bool, optional): Whether to use pre-trained model, default is `True`

    Returns:
    - model (torch.nn.Module): Initialized model
    """

    # resnet series model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # ViT series model
    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_b_32":
        model = models.vit_b_32(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_l_16":
        model = models.vit_l_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_l_32":
        model = models.vit_l_32(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)
    elif model_name == "vit_h_14":
        model = models.vit_h_14(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_class)

    # SiwnTrans series model
    elif model_name == "swin_v2_t":
        model = models.swin_v2_t(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_s":
        model = models.swin_v2_s(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)
    elif model_name == "swin_v2_b":
        model = models.swin_v2_b(pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_class)

    # Mobilenet series model
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_class)
        )
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_class)
        )
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, num_class)
        )

    # EfficientNet series model
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b5":
        model = models.efficientnet_b5(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b6":
        model = models.efficientnet_b6(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )

    # DenseNet series model
    elif model_name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    elif model_name == "densenet169":
        model = models.densenet169(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    elif model_name == "densenet201":
        model = models.densenet201(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_class)

    # VGG series model
    elif model_name == "vgg11":
        model = models.vgg11(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg13":
        model = models.vgg13(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg13_bn":
        model = models.vgg13_bn(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg16_bn":
        model = models.vgg16_bn(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    elif model_name == "vgg19_bn":
        model = models.vgg19_bn(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    # Inception and GoogLeNet
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_class)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # ShuffleNet v2 series model
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "shufflenet_v2_x1_5":
        model = models.shufflenet_v2_x1_5(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "shufflenet_v2_x2_0":
        model = models.shufflenet_v2_x2_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # ResNeXt series model
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "resnext101_32x8d":
        model = models.resnext101_32x8d(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # Wide ResNet series model
    elif model_name == "wide_resnet50_2":
        model = models.wide_resnet50_2(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)
    elif model_name == "wide_resnet101_2":
        model = models.wide_resnet101_2(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_class)

    # MNASNet series model
    elif model_name == "mnasnet0_5":
        model = models.mnasnet0_5(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "mnasnet0_75":
        model = models.mnasnet0_75(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )
    elif model_name == "mnasnet1_3":
        model = models.mnasnet1_3(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, num_class)
        )

    # ConvNeXt series model
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.LayerNorm((768,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1),
            nn.Linear(768, num_class)
        )
    elif model_name == "convnext_small":
        model = models.convnext_small(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.LayerNorm((768,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1),
            nn.Linear(768, num_class)
        )
    elif model_name == "convnext_base":
        model = models.convnext_base(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.LayerNorm((1024,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1),
            nn.Linear(1024, num_class)
        )
    elif model_name == "convnext_large":
        model = models.convnext_large(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.LayerNorm((1536,), eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1),
            nn.Linear(1536, num_class)
        )

    # AlexNet
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(model.classifier[1].in_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class)
        )

    # SqueezeNet series model
    elif model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_class, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_class, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    else:
        raise ValueError("model not supported")

    return model


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
                device=self.parameters['device'],
                batch_size=self.parameters['batch_size'],
                shuffle=self.parameters['shuffle'],
                image_size=self.parameters['image_size'],
                lr=self.parameters['lr'],
            )
        else:
            super().__init__(Basetrainer)

        self.class_idx = self.train_set.dataset.class_to_idx
        if self.save_yaml:
            self.logger.log_with_color(f"Saving yaml file at {self.parameters['save_path']}")

    @property
    def save_yaml(self):

        self.parameters['class_names'] = self.class_idx
        with open(os.path.join(self.save_path, 'config.yaml'), 'w', encoding='utf-8') as file:
            yaml.dump(self.parameters, file, allow_unicode=True)
        return True

    @property
    def train(self):
        num_epochs = self.parameters['num_epochs']

        for epoch in range(num_epochs):
            self.logger.log_with_color(f"Epoch [{epoch + 1}/{num_epochs}] started.")
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in self.train_set:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # forward propagation
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # backward propagation
                loss.backward()
                self.optimizer.step()

                # acc & loss
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            train_loss = running_loss / len(self.train_set)
            train_acc = 100 * correct / total
            self.logger.log_with_color(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
            metrics = self.val
            self.logger.log_with_color(f' Validation Loss: {metrics["total_loss"]:.4f},')
            self.logger.log_with_color(f' Validation Accuracy: {metrics["acc"]:.2f}%,')
            self.logger.log_with_color(f' Validation macro_F1: {metrics["f1"]["macro_f1"]}')
            self.logger.log_with_color(f' Validation micro_F1: {metrics["f1"]["micro_f1"]}')
            self.logger.log_with_color(f' Validation mAP: {metrics["mAP"]["mAP"]}')
            self.logger.log_with_color(f' Validation Top-k Accuracy: {metrics["Top-k"]}')

            self.save_model(metrics, epoch)


class DetTrainer:
    def __init__(self, model_name, dataset_dir):

        self.dataset_dir = dataset_dir

        if model_name == 'yolo':
            self.train = self.yolo_train

    def yolo_train(self, save_dir):
        opt = yolo_init(known=True)  # modify the args in yolo_init if you need to train a custom model
        hyp = opt.hyp
        callbacks = Callbacks()
        device = select_device(opt.device, batch_size=opt.batch_size)
        epochs, batch_size, weights, evolve, data, cfg, noval, nosave, workers, freeze = opt.epochs, opt.batch_size, \
        opt.weights, opt.evolve, opt.data, opt.cfg, opt.noval, opt.nosave, opt.workers, opt.freeze
        if save_dir[-1] != '/':
            # Directories
            save_dir = Path(save_dir)
            w = save_dir / 'weights'  # weights dir
            (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
            last, best = w / 'last.pt', w / 'best.pt'
        else:
            save_dir = Path(save_dir[:-1])
            w = save_dir / 'weights'  # weights dir
            (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
            last, best = w / 'last.pt', w / 'best.pt'

        # Hyperparameters
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
        print(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        opt.hyp = hyp.copy()  # for saving hyps to checkpoints

        # Save run settings
        if not evolve:
            yaml_save(save_dir / 'hyp.yaml', hyp)
            yaml_save(save_dir / 'opt.yaml', vars(opt))

        # Config
        init_seeds(opt.seed + 1, deterministic=True)

        with torch_distributed_zero_first(-1):
            data_dict = check_dataset(data)  # check if None
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = int(data_dict['nc'])  # number of classes
        names = data_dict['names']  # class names

        # Model
        check_suffix(weights, '.pt')  # check weights

        model = DetectionModel(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        amp = False  # check AMP

        # Freeze
        freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze):
                print(f'freezing {k}')
                v.requires_grad = False

        # Image size
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
        optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

        # Scheduler
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        # EMA
        ema = ModelEMA(model)
        # Resume
        best_fitness, start_epoch = 0.0, 0

        # Trainloader
        train_loader, dataset = create_dataloader(path=train_path,
                                                  imgsz=imgsz,
                                                  batch_size=batch_size,
                                                  stride=gs,
                                                  hyp=hyp,
                                                  augment=True,
                                                  cache=None if opt.cache == 'val' else opt.cache,
                                                  rect=opt.rect,
                                                  rank=-1,
                                                  workers=workers,
                                                  image_weights=opt.image_weights,
                                                  quad=opt.quad,
                                                  prefix=colorstr('train: '),
                                                  shuffle=True,
                                                  seed=opt.seed)
        labels = np.concatenate(dataset.labels, 0)
        mlc = int(labels[:, 0].max())  # max label class
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        val_loader = create_dataloader(path=val_path,
                                       imgsz=imgsz,
                                       batch_size=batch_size // 2,
                                       stride=gs,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
        model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

        # Model attributes
        nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = nc  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
        model.names = names

        # Start training
        t0 = time.time()
        nb = len(train_loader)  # number of batches
        nw = max(round(hyp['warmup_epochs'] * nb),
                 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
        # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
        last_opt_step = -1
        maps = np.zeros(nc)  # mAP per class
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        stopper, stop = EarlyStopping(patience=opt.patience), False
        compute_loss = ComputeLoss(model)  # init loss class
        callbacks.run('on_train_start')
        print(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers} dataloader workers\n'
                    f"Logging results to {colorstr('bold', save_dir)}\n"
                    f'Starting training for {epochs} epochs...')

        for epoch in range(start_epoch, epochs):  # epoch --------------------------------------------------------------
            callbacks.run('on_train_epoch_start')
            model.train()

            # Update image weights (optional, single-GPU only)
            if opt.image_weights:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

            # Update mosaic border (optional)
            # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
            # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

            mloss = torch.zeros(3, device=device)  # mean losses

            pbar = enumerate(train_loader)
            print(
                ('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar

            optimizer.zero_grad()
            for i, (
            imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
                callbacks.run('on_train_batch_start')
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Warmup
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi,
                                            [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

                    # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if opt.quad:
                        loss *= 4.

                # Backward
                scaler.scale(loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    last_opt_step = ni

                # Log
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return

            # Scheduler
            lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
            scheduler.step()

            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                stop = stopper(epoch=epoch, fitness=fi)  # early stop check
                if fi > best_fitness:
                    best_fitness = fi
                log_vals = list(mloss) + list(results) + lr
                callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

                # Save model
                if (not nosave) or (final_epoch and not evolve):  # if save
                    ckpt = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'opt': vars(opt),
                        'date': datetime.now().isoformat()}

                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if best_fitness == fi:
                        torch.save(ckpt, best)
                    if opt.save_period > 0 and epoch % opt.save_period == 0:
                        torch.save(ckpt, w / f'epoch{epoch}.pt')
                    del ckpt
                    callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

            # EarlyStopping
            if stop:
                break  # must break all DDP ranks

        print(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        callbacks.run('on_train_end', last, best, epoch, results)

        torch.cuda.empty_cache()
        return results


# for test--------------------------------------------------------------------------------------------------------------
def show_img_in_dataloader(images):
    """Imshow for Tensor."""
    images = images.numpy().transpose((1, 2, 0))
    cv2.imshow('test', images)
    cv2.waitKey(0)