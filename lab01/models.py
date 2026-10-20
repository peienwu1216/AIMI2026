"""PyTorch torchvision ResNet18 and ResNet50 model builders."""
from __future__ import annotations
import os
from pathlib import Path

# Keep pretrained weights with the project. Respect an explicit caller override.
os.environ.setdefault('TORCH_HOME', str(Path(__file__).resolve().parent / '.torch'))

import torch.nn as nn
from torchvision import models

VALID_MODELS = ('resnet18', 'resnet50')


def build_model(name: str, pretrained: bool, trainable_stages='all', dropout=0.0):
    if name not in VALID_MODELS:
        raise ValueError(f'Unknown model: {name}')
    constructors = {'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
                    'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT)}
    constructor, default_weights = constructors[name]
    if not 0 <= dropout < 1:
        raise ValueError('dropout must be in [0, 1)')
    model = constructor(weights=default_weights if pretrained else None)
    # REQUIRED by spec: reinitialize ImageNet 1000-class head for 2 classes.
    in_features = model.fc.in_features
    model.fc = (nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 2))
                if dropout > 0 else nn.Linear(in_features, 2))
    if trainable_stages != 'all':
        if trainable_stages not in ('head', 'layer4'):
            raise ValueError('trainable_stages must be one of all, layer4, head')
        for pname, p in model.named_parameters():
            p.requires_grad = pname.startswith('fc.') or (
                trainable_stages == 'layer4' and pname.startswith('layer4.'))
    return model


def set_frozen_batchnorm_eval(model):
    """Ensure frozen backbone BN stats are not updated during training."""
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.eval()
