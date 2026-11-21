"""Evaluate an image model on a labeled test set.

The test set is either a folder with one sub-folder per class, named exactly as in
model/labels.txt (for example `test/Cornea Abnormal/*.jpg`), or a CSV file with
`path,label` columns (paths relative to the CSV). Images go through the same
preprocessing as the server.

    python scripts/evaluate.py test/ --model model/keras_model.h5 --out report.json

Only use images the model was not trained on, and keep photos of the same person in
one set: otherwise the numbers are too optimistic.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import metrics
from backend.image_predictor import EXPECTED, InvalidImage, file_sha256, preprocess

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_items(source):
    """[(path, class index)] from a class-per-folder tree or a path,label CSV."""
    source = Path(source)
    items = []
    if source.is_dir():
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            if folder.name not in EXPECTED:
                raise SystemExit(f"Unknown class folder: {folder.name!r}")
            for f in sorted(folder.rglob("*")):
                if f.suffix.lower() in EXTENSIONS:
                    items.append((f, EXPECTED.index(folder.name)))
    else:
        with open(source, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["label"] not in EXPECTED:
                    raise SystemExit(f"Unknown label: {row['label']!r}")
                items.append((source.parent / row["path"], EXPECTED.index(row["label"])))
    if not items:
        raise SystemExit("No images found.")
    return items


def predict_all(model, items, batch_size=16):
    """Class probabilities for each readable image; unreadable ones are reported."""
    probs, truth, skipped = [], [], []
    batch, batch_truth = [], []

    def flush():
        if batch:
            out = np.asarray(model(np.concatenate(batch), training=False), dtype=float)
            probs.extend(out)
            truth.extend(batch_truth)
            batch.clear()
            batch_truth.clear()

    for path, label in items:
        try:
            batch.append(preprocess(Path(path).read_bytes()))
            batch_truth.append(label)
        except (InvalidImage, OSError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
        if len(batch) == batch_size:
            flush()
    flush()
    return np.array(probs), np.array(truth, dtype=int), skipped


def build_report(probs, truth, skipped, model_path=None):
    ece, bins = metrics.expected_calibration_error(probs, truth)
    return {
        "model": str(model_path) if model_path else None,
        "model_sha256": file_sha256(model_path) if model_path else None,
        "images": int(len(truth)),
        "skipped": skipped,
        "accuracy": float((probs.argmax(1) == truth).mean()),
        "per_class": metrics.per_class(probs, truth, EXPECTED),
        "confusion_matrix": metrics.confusion_matrix(truth, probs.argmax(1), len(EXPECTED)).tolist(),
        "ece": ece,
        "calibration_bins": bins,
        "threshold_sweep": metrics.threshold_sweep(probs, truth, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    }


def no_nan(value):
    """JSON has no NaN: write undefined metrics (e.g. a class with no images) as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [no_nan(v) for v in value]
    return value


def fmt(value):
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.2f}"


def summary(report):
    lines = [
        f"Images: {report['images']} (skipped {len(report['skipped'])}), "
        f"top-1 accuracy {fmt(report['accuracy'])}, ECE {fmt(report['ece'])}",
        "",
        "| Class | n | Sensitivity (95% CI) | Specificity (95% CI) | PPV | AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["per_class"]:
        lo, hi = r["sensitivity_ci"]
        slo, shi = r["specificity_ci"]
        lines.append(
            f"| {r['class']} | {r['n']} | {fmt(r['sensitivity'])} ({fmt(lo)}-{fmt(hi)}) | "
            f"{fmt(r['specificity'])} ({fmt(slo)}-{fmt(shi)}) | {fmt(r['ppv'])} | {fmt(r['auroc'])} |"
        )
    lines += ["", "| Top-1 threshold | Images accepted | Accuracy of accepted |", "|---|---|---|"]
    for r in report["threshold_sweep"]:
        lines.append(f"| {r['threshold']:.1f} | {fmt(r['coverage'])} | {fmt(r['accuracy'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test_set", help="class-per-folder directory or path,label CSV")
    parser.add_argument("--model", type=Path, default=Path("model/keras_model.h5"))
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    import tensorflow as tf

    model = tf.keras.models.load_model(args.model, compile=False)
    if tuple(model.output_shape) != (None, len(EXPECTED)):
        raise SystemExit(f"Model output must be (None, {len(EXPECTED)}).")
    probs, truth, skipped = predict_all(model, load_items(args.test_set))
    if not len(truth):
        raise SystemExit("No readable images.")
    report = build_report(probs, truth, skipped, args.model)
    print(summary(report))
    if args.out:
        args.out.write_text(json.dumps(no_nan(report), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
