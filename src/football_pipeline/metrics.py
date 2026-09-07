"""Metrics for the three-way result target. Classes are always 0, 1, 2."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

ALL_CLASSES = np.array([0, 1, 2])
CLASS_NAMES = {0: "away", 1: "draw", 2: "home"}


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width reliability bins. Used for diagnostics, not for threshold search."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if not mask.any():
            continue
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "mean_observed": float(y[mask].mean()),
            }
        )
    return out


def pad_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map a model's probability columns onto [away, draw, home]."""
    out = np.zeros((proba.shape[0], 3), dtype=float)
    for index, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int <= 2:
            out[:, cls_int] = proba[:, index]
    row_sums = out.sum(axis=1, keepdims=True)
    missing = row_sums.ravel() <= 0
    out[missing] = 1.0 / 3.0
    row_sums = out.sum(axis=1, keepdims=True)
    return out / row_sums


def evaluate_split(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
    if y_pred is None:
        y_pred = proba.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ALL_CLASSES,
        zero_division=0,
        average=None,
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=ALL_CLASSES,
        zero_division=0,
        average="macro",
    )
    onehot = np.eye(3)[np.asarray(y_true, dtype=int)]
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    p_draw = proba[:, 1]
    pred = y_pred
    draw_mask = np.asarray(y_true) == 1
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, proba, labels=ALL_CLASSES)),
        "brier": brier,
        "f1_macro": float(f1_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "precision_by_class": {CLASS_NAMES[i]: float(precision[i]) for i in ALL_CLASSES},
        "recall_by_class": {CLASS_NAMES[i]: float(recall[i]) for i in ALL_CLASSES},
        "f1_by_class": {CLASS_NAMES[i]: float(f1[i]) for i in ALL_CLASSES},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ALL_CLASSES).tolist(),
        "confusion_matrix_labels": ["away", "draw", "home"],
        "draw_proba": {
            "mean": float(p_draw.mean()),
            "median": float(np.median(p_draw)),
            "min": float(p_draw.min()),
            "max": float(p_draw.max()),
            "pct_argmax_draw": float((pred == 1).mean()),
            "mean_when_true_draw": float(p_draw[draw_mask].mean()) if draw_mask.any() else None,
            "mean_when_not_draw": float(p_draw[~draw_mask].mean()) if (~draw_mask).any() else None,
        },
        "draw_calibration": reliability_bins((np.asarray(y_true) == 1).astype(float), p_draw),
    }


def uniform_proba(n: int) -> np.ndarray:
    return np.full((n, 3), 1.0 / 3.0)


def one_hot_proba(n: int, cls: int) -> np.ndarray:
    out = np.zeros((n, 3), dtype=float)
    out[:, cls] = 1.0
    return out


def empirical_proba(n: int, y_train: np.ndarray) -> np.ndarray:
    counts = np.array([(y_train == c).sum() for c in ALL_CLASSES], dtype=float)
    if counts.sum() == 0:
        return uniform_proba(n)
    prior = counts / counts.sum()
    return np.tile(prior, (n, 1))
