"""ShallowConvNet (Schirrmeister et al., 2017), an extra bonus model.

The network is a learnable version of the filter-bank CSP pipeline:

    temporal conv   -> band-pass filtering
    spatial conv    -> CSP-style spatial filtering
    square          -> signal power
    average pooling -> power averaged over a time window
    log             -> log band power

The paper uses (1, 25) kernels and (1, 75) / stride 15 pooling at 250 Hz. This
dataset is sampled at 125 Hz, so those lengths are halved to keep the same
durations in seconds, which gives (1, 13) and (1, 37) / stride 7.
"""

from __future__ import annotations

import torch
from torch import nn


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        dropout: float = 0.5,
        num_classes: int = 2,
        input_channels: int = 2,
        input_samples: int = 750,
        filters: int = 40,
        temporal_kernel: int = 13,
        pool_kernel: int = 37,
        pool_stride: int = 7,
        log_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.log_floor = log_floor
        self.temporal = nn.Conv2d(1, filters, kernel_size=(1, temporal_kernel))
        self.spatial = nn.Conv2d(
            filters, filters, kernel_size=(input_channels, 1), bias=False
        )
        self.batch_norm = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.1)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(p=dropout)

        conv_samples = input_samples - temporal_kernel + 1
        pooled_samples = (conv_samples - pool_kernel) // pool_stride + 1
        if pooled_samples < 1:
            raise ValueError("input_samples is too short for ShallowConvNet")
        self.feature_dim = filters * pooled_samples
        self.classify = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(self.spatial(self.temporal(x)))
        x = self.pool(x.square())
        x = torch.log(x.clamp_min(self.log_floor))
        return torch.flatten(self.dropout(x), start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.forward_features(x))
