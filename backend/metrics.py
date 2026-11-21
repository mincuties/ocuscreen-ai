"""Evaluation metrics for the single-label image classifier, numpy only.

Used by scripts/evaluate.py. Kept free of TensorFlow so it can be unit tested.
"""

import math

import numpy as np


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval for a proportion; (nan, nan) when total is 0."""
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def auroc(scores, positives):
    """Area under the ROC curve (Mann-Whitney U, ties counted as half); nan if one class."""
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion_matrix(truth, predicted, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[t, p] += 1
    return matrix


def per_class(probs, truth, names):
    """Sensitivity, specificity and PPV of the top-1 decision, plus one-vs-rest AUROC."""
    probs = np.asarray(probs, dtype=float)
    truth = np.asarray(truth, dtype=int)
    predicted = probs.argmax(1)
    rows = []
    for k, name in enumerate(names):
        actual, called = truth == k, predicted == k
        tp, fn = int((actual & called).sum()), int((actual & ~called).sum())
        fp, tn = int((~actual & called).sum()), int((~actual & ~called).sum())
        rows.append(
            {
                "class": name,
                "n": int(actual.sum()),
                "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
                "sensitivity_ci": wilson_interval(tp, tp + fn),
                "specificity": tn / (tn + fp) if tn + fp else math.nan,
                "specificity_ci": wilson_interval(tn, tn + fp),
                "ppv": tp / (tp + fp) if tp + fp else math.nan,
                "auroc": auroc(probs[:, k], actual),
            }
        )
    return rows


def expected_calibration_error(probs, truth, bins=10):
    """ECE of the top-1 confidence over equal-width bins, plus the bins themselves."""
    probs = np.asarray(probs, dtype=float)
    truth = np.asarray(truth, dtype=int)
    confidence = probs.max(1)
    correct = probs.argmax(1) == truth
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi) if lo > 0 else confidence <= hi
        if not inside.any():
            continue
        acc, conf = float(correct[inside].mean()), float(confidence[inside].mean())
        ece += inside.mean() * abs(acc - conf)
        table.append({"range": (float(lo), float(hi)), "n": int(inside.sum()),
                      "accuracy": acc, "confidence": conf})
    return float(ece), table


def threshold_sweep(probs, truth, thresholds):
    """For each top-1 threshold: share of images accepted and accuracy among them."""
    probs = np.asarray(probs, dtype=float)
    truth = np.asarray(truth, dtype=int)
    confidence, correct = probs.max(1), probs.argmax(1) == truth
    rows = []
    for t in thresholds:
        kept = confidence >= t
        rows.append({"threshold": float(t), "coverage": float(kept.mean()),
                     "accuracy": float(correct[kept].mean()) if kept.any() else math.nan})
    return rows
