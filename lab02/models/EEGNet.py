"""EEG models required by the Lab02 specification."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def make_activation(
    name: str,
    *,
    elu_alpha: float = 1.0,
    leaky_slope: float = 0.01,
) -> nn.Module:
    """Create a fresh activation module for one network block."""
    factories: dict[str, Callable[[], nn.Module]] = {
        "elu": lambda: nn.ELU(alpha=elu_alpha),
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=leaky_slope),
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "selu": nn.SELU,
        "mish": nn.Mish,
    }
    key = name.lower()
    if key not in factories:
        raise ValueError(f"Unsupported activation {name!r}; choose from {sorted(factories)}")
    return factories[key]()


class EEGNet(nn.Module):
    """Course EEGNet plus an optional true depthwise-separable second block.

    The course screenshot labels a single dense temporal convolution as
    ``separableConv``. ``separable=False`` reproduces that screenshot exactly;
    ``separable=True`` uses depthwise temporal convolution followed by a 1x1
    pointwise convolution, which is the mathematical separable operation.
    """

    def __init__(
        self,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        dropout: float = 0.25,
        F1: int = 16,
        D: int = 2,
        F2: int = 32,
        kernel_length: int = 51,
        sep_kernel: int = 15,
        separable: bool = False,
        num_classes: int = 2,
        input_samples: int = 750,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if kernel_length % 2 == 0 or sep_kernel % 2 == 0:
            raise ValueError("Temporal kernels must be odd so same-padding is exact")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if D < 1 or F1 < 1 or F2 < 1:
            raise ValueError("F1, D, and F2 must be positive")

        channels = F1 * D
        activation_kwargs = {"elu_alpha": elu_alpha, "leaky_slope": leaky_slope}

        self.firstconv = nn.Sequential(
            nn.Conv2d(
                1,
                F1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, channels, kernel_size=(2, 1), groups=F1, bias=False),
            nn.BatchNorm2d(channels),
            make_activation(activation, **activation_kwargs),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )

        if separable:
            temporal_conv: nn.Module = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, sep_kernel),
                    padding=(0, sep_kernel // 2),
                    groups=channels,
                    bias=False,
                ),
                nn.Conv2d(channels, F2, kernel_size=(1, 1), bias=False),
            )
        else:
            temporal_conv = nn.Conv2d(
                channels,
                F2,
                kernel_size=(1, sep_kernel),
                padding=(0, sep_kernel // 2),
                bias=False,
            )

        self.separableConv = nn.Sequential(
            temporal_conv,
            nn.BatchNorm2d(F2),
            make_activation(activation, **activation_kwargs),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )

        pooled_samples = (input_samples // 4) // 8
        if pooled_samples < 1:
            raise ValueError("input_samples is too short for the two pooling layers")
        self.feature_dim = F2 * pooled_samples
        self.classify = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Expected {self.feature_dim} flattened features, got {features.shape[1]}; "
                "check input shape and input_samples"
            )
        return self.classify(features)


class DeepConvNet(nn.Module):
    """DeepConvNet bonus architecture from the course table (C=2, T=750)."""

    def __init__(
        self,
        activation: str = "elu",
        elu_alpha: float = 1.0,
        dropout: float = 0.5,
        num_classes: int = 2,
        input_samples: int = 750,
        leaky_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        activation_kwargs = {"elu_alpha": elu_alpha, "leaky_slope": leaky_slope}

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout),
        )
        self.block2 = self._later_block(25, 50, activation, activation_kwargs, dropout)
        self.block3 = self._later_block(50, 100, activation, activation_kwargs, dropout)
        self.block4 = self._later_block(100, 200, activation, activation_kwargs, dropout)

        temporal_size = input_samples
        for _ in range(4):
            temporal_size = (temporal_size - 5 + 1) // 2
        if temporal_size < 1:
            raise ValueError("input_samples is too short for DeepConvNet")
        self.feature_dim = 200 * temporal_size
        self.classify = nn.Linear(self.feature_dim, num_classes)

    @staticmethod
    def _later_block(
        in_channels: int,
        out_channels: int,
        activation: str,
        activation_kwargs: dict[str, float],
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5)),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            make_activation(activation, **activation_kwargs),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                f"Expected {self.feature_dim} flattened features, got {features.shape[1]}; "
                "check input shape and input_samples"
            )
        return self.classify(features)
