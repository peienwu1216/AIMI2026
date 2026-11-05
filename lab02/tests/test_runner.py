import pytest
import torch

from main import MODEL_BUILDERS, build_model, make_loaders, parse_args


def test_all_models_are_registered() -> None:
    assert set(MODEL_BUILDERS) == {"eegnet", "deepconvnet", "shallowconvnet"}
    args = parse_args(["--model", "shallowconvnet"])
    assert build_model(args)(torch.randn(2, 1, 2, 750)).shape == (2, 2)


def test_model_specific_dropout_defaults() -> None:
    assert parse_args(["--model", "eegnet"]).dropout == 0.25
    assert parse_args(["--model", "deepconvnet"]).dropout == 0.5
    assert parse_args(["--model", "shallowconvnet"]).dropout == 0.5


def test_zero_validation_uses_all_training_samples() -> None:
    args = parse_args(
        [
            "--val-ratio",
            "0",
            "--select-by",
            "test_accuracy",
            "--smoke-samples",
            "32",
        ]
    )
    args.device_resolved = torch.device("cpu")
    train_loader, val_loader, test_loader = make_loaders(args)
    assert len(train_loader.dataset) == 32
    assert val_loader is None
    assert len(test_loader.dataset) == 32


def test_zero_validation_requires_test_selection() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--val-ratio", "0"])


def test_test_selection_requires_epochwise_test() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--select-by", "test_accuracy", "--no-test-each-epoch"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--num-epochs", "0"],
        ["--batch-size", "0"],
        ["--lr", "0"],
        ["--smoke-samples", "-1"],
        ["--patience", "-1"],
    ],
)
def test_invalid_numeric_ranges(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(arguments)
