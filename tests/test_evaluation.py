import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backend import metrics
import backend.image_predictor as ip

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


def test_binary_rates_and_average_precision():
    r = metrics.binary_rates([0.9, 0.6, 0.4, 0.2, 0.7], [1, 1, 1, 0, 0], 0.5)
    assert r["n_pos"] == 3 and r["sensitivity"] == pytest.approx(2 / 3)
    assert r["specificity"] == pytest.approx(0.5) and r["ppv"] == pytest.approx(2 / 3)
    assert metrics.average_precision([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert math.isnan(metrics.average_precision([0.1], [0]))


def test_binary_ece():
    # Bin 0.9-1.0: 2 images at 0.95, one positive -> |0.95 - 0.5|.
    assert metrics.binary_ece([0.95, 0.95], [1, 0]) == pytest.approx(0.45)
    assert metrics.binary_ece([0.05, 0.95], [0, 1]) == pytest.approx(0.05)


def save(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color).save(path)


def test_load_items_from_dataset_csv(tmp_path):
    save(tmp_path / "images" / "a.png", "white")
    save(tmp_path / "images" / "b.png", "black")
    header = "file,split,redness,discharge,normal\n"
    (tmp_path / "labels.csv").write_text(
        header + "images/a.png,test,1,1,0\nimages/b.png,train,0,0,1\n", encoding="utf-8"
    )
    items = evaluate.load_items(tmp_path / "labels.csv")
    assert items == [(tmp_path / "images" / "a.png", {"Redness", "Discharge"})]
    assert len(evaluate.load_items(tmp_path / "labels.csv", "all")) == 2


def test_load_items_from_simple_csv(tmp_path):
    (tmp_path / "set.csv").write_text("path,labels\na.png,Redness;Swelling\nb.png,\n", encoding="utf-8")
    assert evaluate.load_items(tmp_path / "set.csv") == [
        (tmp_path / "a.png", {"Redness", "Swelling"}), (tmp_path / "b.png", set())
    ]
    (tmp_path / "bad.csv").write_text("path,labels\na.png,Pink Eye\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate.load_items(tmp_path / "bad.csv")


def test_evaluate_end_to_end_with_fake_model(tmp_path, fake_model):
    fake_model({"Redness": 0.9, "Normal": 0.1})
    save(tmp_path / "a.png", "white")
    save(tmp_path / "b.png", "black")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    (tmp_path / "set.csv").write_text(
        "path,labels\na.png,Redness\nb.png,\nbroken.png,Redness\n", encoding="utf-8"
    )
    session, card = ip.load_artifacts()
    items = evaluate.load_items(tmp_path / "set.csv")
    probs, truth, skipped, flags = evaluate.predict_all(session, card, items)
    report = evaluate.build_report(card, probs, truth, skipped, flags)
    assert report["images"] == 2 and len(report["skipped"]) == 1
    rows = {r["label"]: r for r in report["labels"]}
    # The fake model calls Redness on every photo.
    assert rows["Redness"]["sensitivity"] == 1.0 and rows["Redness"]["specificity"] == 0.0
    assert rows["Normal"]["n_pos"] == 1 and report["normal_with_a_sign"] == 1.0
    assert report["eye_gate_pass"] == 1.0
    json.dumps(evaluate.no_nan(report), allow_nan=False)
    text = evaluate.summary(report)
    assert "| Redness | yes | 1 |" in text and "| Yellow Eye | no | 0 |" in text


def test_centre_crop():
    image = Image.new("RGB", (200, 100))
    assert evaluate.centre_crop(image, 0.5).size == (50, 50)
    assert evaluate.centre_crop(image, 1.0) is image
