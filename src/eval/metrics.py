"""Shared metric helpers."""

from __future__ import annotations


def compute_binary_metrics(y_true, y_pred):
    tp = sum(pred == 1 and true == 1 for true, pred in zip(y_true, y_pred))
    fp = sum(pred == 1 and true == 0 for true, pred in zip(y_true, y_pred))
    tn = sum(pred == 0 and true == 0 for true, pred in zip(y_true, y_pred))
    fn = sum(pred == 0 and true == 1 for true, pred in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
