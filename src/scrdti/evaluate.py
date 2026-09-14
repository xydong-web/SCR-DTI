from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    probabilities = np.asarray(probabilities).reshape(-1).astype(np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have matching shapes")
    if labels.size == 0:
        raise ValueError("cannot evaluate an empty prediction set")
    predicted = (probabilities >= threshold).astype(np.int64)
    metrics = {
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
    }
    metrics["auroc"] = (
        float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else math.nan
    )
    return metrics


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    return float(np.mean((probabilities - labels) ** 2))


def negative_log_likelihood(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    eps: float = 1e-7,
) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), eps, 1 - eps)
    return float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(labels), 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        error += float(mask.sum()) / total * abs(confidence - accuracy)
    return float(error)

