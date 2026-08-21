"""
Evaluation metrics. Primary metric: macro F1-score.
"""

import torch
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Calculate classification metrics between predictions and ground truth labels.
    Inputs are converted to NumPy for compatibility with scikit-learn metrics.
    Arguments 'preds' and 'labels' are torch Tensors of equal shape.
    Returns a dictionary containing 'f1_macro' and 'accuracy' scores.
    """
    p = preds.cpu().numpy()
    l = labels.cpu().numpy()
    return {
        "f1_macro": f1_score(l, p, average="macro"),
        "accuracy": accuracy_score(l, p),
    }
