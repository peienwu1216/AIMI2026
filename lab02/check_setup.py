#!/usr/bin/env python3
"""Fast, read-only validation of the Lab02 environment and dataset."""

import platform

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

from dataloader import read_bci_data


def available_device() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    train_x, train_y, test_x, test_y = read_bci_data()

    assert train_x.shape == (1080, 1, 2, 750), train_x.shape
    assert test_x.shape == (1080, 1, 2, 750), test_x.shape
    assert train_y.shape == (1080,), train_y.shape
    assert test_y.shape == (1080,), test_y.shape
    assert set(np.unique(train_y)) == {0.0, 1.0}
    assert set(np.unique(test_y)) == {0.0, 1.0}
    assert np.isfinite(train_x).all()
    assert np.isfinite(test_x).all()
    assert np.bincount(train_y.astype(np.int64)).tolist() == [540, 540]
    assert np.bincount(test_y.astype(np.int64)).tolist() == [540, 540]

    print(f"Python:     {platform.python_version()}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"NumPy:      {np.__version__}")
    print(f"Pandas:     {pd.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Device:     {available_device()}")
    print(f"Train:      {train_x.shape}, labels={np.bincount(train_y.astype(np.int64)).tolist()}")
    print(f"Test:       {test_x.shape}, labels={np.bincount(test_y.astype(np.int64)).tolist()}")
    print("READY")


if __name__ == "__main__":
    main()
