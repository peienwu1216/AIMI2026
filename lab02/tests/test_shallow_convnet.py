import pytest
import torch

from models.ShallowConvNet import ShallowConvNet


def test_shape_feature_size_and_parameter_count() -> None:
    model = ShallowConvNet()
    assert model(torch.randn(3, 1, 2, 750)).shape == (3, 2)
    # (750 - 13 + 1) = 738 samples, then (738 - 37) // 7 + 1 = 101 pooled steps.
    assert model.feature_dim == 40 * 101
    # temporal 40*13+40, spatial 40*40*2, batch norm 2*40, dense 4040*2+2.
    assert sum(p.numel() for p in model.parameters()) == 560 + 3200 + 80 + 8082


def test_backward_reaches_every_parameter() -> None:
    model = ShallowConvNet()
    model(torch.randn(4, 1, 2, 750)).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_log_stays_finite_on_silent_input() -> None:
    model = ShallowConvNet().eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(2, 1, 2, 750))).all()


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        ShallowConvNet(dropout=1.0)
    with pytest.raises(ValueError):
        ShallowConvNet(input_samples=20)
