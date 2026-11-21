import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import metrics
from backend.image_predictor import EXPECTED

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402


def test_wilson_interval_known_values():
    lo, hi = metrics.wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert all(math.isnan(v) for v in metrics.wilson_interval(0, 0))


def test_auroc_perfect_random_and_ties():
    assert metrics.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert metrics.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert metrics.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert math.isnan(metrics.auroc([0.1, 0.2], [1, 1]))


def test_per_class_counts():
    # Three images of class 0, two of class 1; one class-0 image is called class 1.
    probs = np.zeros((5, len(EXPECTED)))
    probs[[0, 1], 0] = 0.9
    probs[[2, 3, 4], 1] = 0.9
    truth = [0, 0, 0, 1, 1]
    rows = metrics.per_class(probs, truth, EXPECTED)
    assert rows[0]["n"] == 3 and rows[0]["sensitivity"] == pytest.approx(2 / 3)
    assert rows[1]["sensitivity"] == 1.0 and rows[1]["specificity"] == pytest.approx(2 / 3)
    assert rows[1]["ppv"] == pytest.approx(2 / 3)
    assert math.isnan(rows[5]["sensitivity"])  # no images of that class


def test_calibration_and_threshold_sweep():
    probs = np.zeros((4, len(EXPECTED)))
    probs[:, 0] = [0.95, 0.95, 0.55, 0.55]
    truth = [0, 0, 0, 1]
    ece, bins = metrics.expected_calibration_error(probs, truth)
    # Bin 0.9-1.0: accuracy 1.0 vs 0.95; bin 0.5-0.6: accuracy 0.5 vs 0.55.
    assert ece == pytest.approx(0.5 * 0.05 + 0.5 * 0.05)
    assert [b["n"] for b in bins] == [2, 2]
    sweep = {r["threshold"]: r for r in metrics.threshold_sweep(probs, truth, [0.5, 0.9])}
    assert sweep[0.5]["coverage"] == 1.0 and sweep[0.5]["accuracy"] == 0.75
    assert sweep[0.9]["coverage"] == 0.5 and sweep[0.9]["accuracy"] == 1.0


def save(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color).save(path)


def test_evaluate_end_to_end_with_fake_model(tmp_path):
    save(tmp_path / "Normal" / "a.png", "white")
    save(tmp_path / "Redness" / "b.png", "black")
    (tmp_path / "Redness" / "broken.png").write_bytes(b"not an image")

    class Model:
        """Calls white images Normal and everything else Redness."""

        def __call__(self, batch, training=False):
            out = np.zeros((len(batch), len(EXPECTED)))
            white = batch.mean(axis=(1, 2, 3)) > 0
            out[white, 0] = 1
            out[~white, 1] = 1
            return out

    probs, truth, skipped = evaluate.predict_all(Model(), evaluate.load_items(tmp_path))
    report = evaluate.build_report(probs, truth, skipped)
    assert report["images"] == 2 and report["accuracy"] == 1.0
    assert len(report["skipped"]) == 1
    json.dumps(evaluate.no_nan(report), allow_nan=False)
    assert "| Normal | 1 |" in evaluate.summary(report)


def test_load_items_rejects_unknown_class(tmp_path):
    save(tmp_path / "Pink Eye" / "a.png", "white")
    with pytest.raises(SystemExit):
        evaluate.load_items(tmp_path)


def test_load_items_from_csv(tmp_path):
    save(tmp_path / "img" / "a.png", "white")
    (tmp_path / "set.csv").write_text("path,label\nimg/a.png,Normal\n", encoding="utf-8")
    assert evaluate.load_items(tmp_path / "set.csv") == [(tmp_path / "img" / "a.png", 0)]
