"""Evaluation metrics for the multi-label image model, numpy only.

Used by scripts/evaluate.py and training/fit.py. Kept free of model libraries so it can
be unit tested.
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


def average_precision(scores, positives):
    """Area under the precision-recall curve (mean precision at each positive); nan if none."""
    positives = np.asarray(positives, dtype=bool)
    if positives.sum() == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    hits = positives[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def binary_rates(probs, positives, threshold):
    """Sensitivity, specificity (with 95% Wilson intervals) and PPV at one threshold."""
    called = np.asarray(probs, dtype=float) >= threshold
    actual = np.asarray(positives, dtype=bool)
    tp, fn = int((called & actual).sum()), int((~called & actual).sum())
    tn, fp = int((~called & ~actual).sum()), int((called & ~actual).sum())
    return {
        "n_pos": tp + fn,
        "n_neg": tn + fp,
        "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
        "sensitivity_ci": wilson_interval(tp, tp + fn),
        "specificity": tn / (tn + fp) if tn + fp else math.nan,
        "specificity_ci": wilson_interval(tn, tn + fp),
        "ppv": tp / (tp + fp) if tp + fp else math.nan,
    }


def binary_ece(probs, positives, bins=10):
    """Expected calibration error of one label's probabilities over equal-width bins."""
    probs = np.asarray(probs, dtype=float)
    actual = np.asarray(positives, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (probs > lo) & (probs <= hi) if lo > 0 else probs <= hi
        if inside.any():
            ece += inside.mean() * abs(probs[inside].mean() - actual[inside].mean())
    return float(ece)
