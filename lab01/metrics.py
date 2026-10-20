"""Consistent positive class = 1 (PNEUMONIA)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def summarize(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) != len(y_pred) or not len(y_true):
        raise ValueError('Nonempty equal-length predictions and labels required')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'specificity': float(specificity),
        'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': cm.tolist(),  # [[TN,FP],[FN,TP]]; rows=true, cols=pred
        'support': int(len(y_true)),
    }
