import pytest
import torch

from models import DeepConvNet, EEGNet, make_activation


@pytest.mark.parametrize(
    "activation",
    ["elu", "relu", "leaky_relu", "gelu", "silu", "selu", "mish"],
)
def test_eegnet_activations_forward_backward(activation: str) -> None:
    model = EEGNet(activation=activation)
    inputs = torch.randn(4, 1, 2, 750)
    logits = model(inputs)
    assert logits.shape == (4, 2)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_course_eegnet_shape_and_parameter_count() -> None:
    model = EEGNet()
    assert model(torch.randn(3, 1, 2, 750)).shape == (3, 2)
    assert model.feature_dim == 736
    assert sum(parameter.numel() for parameter in model.parameters()) == 17_874


def test_true_separable_eegnet_is_smaller() -> None:
    course_model = EEGNet(separable=False)
    separable_model = EEGNet(separable=True)
    assert separable_model(torch.randn(2, 1, 2, 750)).shape == (2, 2)
    assert sum(parameter.numel() for parameter in separable_model.parameters()) == 4_018
    assert sum(parameter.numel() for parameter in separable_model.parameters()) < sum(
        parameter.numel() for parameter in course_model.parameters()
    )


def test_deepconvnet_shape_parameter_count_and_backward() -> None:
    model = DeepConvNet()
    logits = model(torch.randn(2, 1, 2, 750))
    assert logits.shape == (2, 2)
    assert model.feature_dim == 8_600
    assert sum(parameter.numel() for parameter in model.parameters()) == 150_977
    logits.mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_no_softmax_is_applied() -> None:
    logits = EEGNet()(torch.randn(8, 1, 2, 750))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8), atol=1e-5)


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        EEGNet(kernel_length=50)
    with pytest.raises(ValueError):
        EEGNet(dropout=1.0)
    with pytest.raises(ValueError):
        make_activation("not-an-activation")
