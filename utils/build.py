"""build from CFG
"""
import logging
import os

import timm
import torch
import yaml

DefaultConfig = '../configs/config.yaml'


def is_supported_model(name: str) -> bool:
    """True if `name` is a valid timm model. Classification models are built
    with `timm` (1000+ architectures); use any name from `timm.list_models()`."""
    return isinstance(name, str) and timm.is_model(name.strip())


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
    if not is_supported_model(opt['model']):
        raise ValueError(
            f"Model '{opt.get('model')}' is not a valid timm model. "
            f"Pick a name from timm.list_models() (e.g. resnet50, "
            f"vit_base_patch16_224, convnext_tiny, swinv2_tiny_window8_256)."
        )
    if not isinstance(opt['num_classes'], int):
        raise ValueError("The number of classes must be an integer")
    if opt['weights'] is None or not os.path.exists(opt['weights']):
        logging.info("No fine-tune checkpoint specified; backbone init follows the `pretrained` flag")
    if opt['device'] != 'cpu' and not torch.cuda.is_available():
        logging.info("CUDA is not available, using CPU instead")
        opt['device'] = "cpu"
    return opt


def build_from_cfg(cfg: str = DefaultConfig):
    if cfg != DefaultConfig:
        # check_cfg validates AND normalises (e.g. device->cpu fallback) and
        # returns the resulting opt; reloading a fresh copy here would silently
        # discard those normalisations.
        opt = check_cfg(cfg)
        logging.info("Using custom config: {}".format(cfg))
        return opt

    return yaml.load(open(cfg, 'r', encoding='utf-8'), Loader=yaml.FullLoader)