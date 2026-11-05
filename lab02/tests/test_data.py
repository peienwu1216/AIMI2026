import numpy as np

from dataloader import read_bci_data


def test_course_data_shapes_labels_balance_and_finiteness() -> None:
    train_x, train_y, test_x, test_y = read_bci_data()
    assert train_x.shape == (1080, 1, 2, 750)
    assert test_x.shape == (1080, 1, 2, 750)
    assert np.bincount(train_y.astype(np.int64)).tolist() == [540, 540]
    assert np.bincount(test_y.astype(np.int64)).tolist() == [540, 540]
    assert np.isfinite(train_x).all()
    assert np.isfinite(test_x).all()
