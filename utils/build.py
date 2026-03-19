"""build from CFG
"""
import logging
import os

import torch
import yaml

DefaultConfig = '../configs/config.yaml'
Model_list = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32", "vit_h_14",
    "swin_v2_t", "swin_v2_s", "swin_v2_b", "mobilenet_v3_small",
    "mobilenet_v3_large"]


def check_cfg(cfg: str):

    opt = yaml.load(open(cfg, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    if len(opt['class_names']) != opt['num_classes']:
        raise ValueError("The number of classes does not match the number of class names")
    if not os.path.exists(opt['train']):
        raise ValueError("Training data path does not exist: {}".format(opt['train']))
    if not os.path.exists(opt['val']):
        raise ValueError("Validation data path does not exist: {}".format(opt['val']))
    if not os.path.exists(opt['save_path']):
        os.makedirs(opt['save_path'], exist_ok=True)
        logging.info("Save path does not exist, created: {}".format(opt['save_path']))
    if not isinstance(opt['model'], str) or opt['model'].lower() not in Model_list:
        raise ValueError("The model you specified is not available")
    if not isinstance(opt['num_classes'], int):
        raise ValueError("The number of classes must be an integer")
    if opt['weights'] == None or not os.path.exists(opt['weights']):
        logging.info("No pretrained weights specified, training from scratch")
        opt['pretrained'] = False
    if opt['device'] != 'cpu' and not torch.cuda.is_available():
        logging.info("CUDA is not available, using CPU instead")
        opt['device'] = "cpu"
    return True


def build_from_cfg(cfg: str = DefaultConfig):
    if cfg != DefaultConfig:
        if not check_cfg(cfg):
            raise ValueError("Invalid config file: {}".format(cfg))
        logging.info("Using custom config: {}".format(cfg))
    opt = yaml.load(open(cfg, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

    return opt