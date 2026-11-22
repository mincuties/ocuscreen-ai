"""Evaluate the ONNX image model on a labeled test set, label by label.

The test set is the dataset's labels.csv (one 0/1 column per finding, plus `file` and
`split`; only `split == test` rows are used unless --split all) or a CSV with
`path,labels` columns where `labels` lists sign names separated by `;` (empty means
Normal). Paths are relative to the CSV. Images go through the server's own
preprocessing and model session, and are scored at the thresholds in model_card.json.

    python scripts/evaluate.py "C:/data/_clean_v1/labels.csv" --out report.json
    python scripts/evaluate.py test.csv --crop 1.0,0.7,0.5

Only use images the model was not trained on, and keep photos of the same person in
one set: otherwise the numbers are too optimistic.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import metrics
from backend import image_predictor as ip
from training.common import NORMAL as NORMAL_COLUMN, SIGNS


def load_items(source, split="test"):
    """[(path, set of sign names)]; Normal is the empty set."""
    source = Path(source)
    items = []
    with open(source, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        for row in reader:
            if "labels" in fields:
                names = {n.strip() for n in row["labels"].split(";") if n.strip()}
                unknown = names - set(ip.KB["image_signs"])
                if unknown:
                    raise SystemExit(f"Unknown labels: {sorted(unknown)}")
                items.append((source.parent / row["path"], names - {ip.NORMAL}))
            elif "file" in fields:
                if split != "all" and row.get("split") != split:
                    continue
                names = {SIGNS[c] for c in SIGNS if c in row and row[c] == "1"}
                items.append((source.parent / row["file"], names))
            else:
                raise SystemExit("CSV needs path,labels columns or a dataset labels.csv")
    if not items:
        raise SystemExit("No images found.")
    return items


def centre_crop(image, fraction):
    if fraction >= 1:
        return image
    w, h = image.size
    side = int(min(w, h) * fraction)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def predict_all(session, card, items, fraction=1.0):
    """Calibrated probabilities (images x labels), truth sets, skipped files, quality flags."""
    probs, truth, skipped, flags = [], [], [], []
    for path, names in items:
        try:
            image = ip.decode(Path(path).read_bytes())
        except (ip.InvalidImage, OSError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        image = centre_crop(image, fraction)
        logits, _ = ip.run_model(session, card, ip.to_batch(image, card["input"]))
        probs.append(ip.calibrated(card, logits))
        truth.append(names)
        flags.append(ip.measure_quality(image)["flags"])
    return np.array(probs), truth, skipped, flags


def build_report(card, probs, truth, skipped, flags):
    labels = [l for l in card["labels"] if l["name"] != ip.GATE]
    rows = []
    for label in labels:
        p = probs[:, label["index"]]
        if label["name"] == ip.NORMAL:
            y = np.array([not names for names in truth])
        else:
            y = np.array([label["name"] in names for names in truth])
        rows.append({
            "label": label["name"],
            "published": label["published"],
            "threshold": label["threshold"],
            **metrics.binary_rates(p, y, label["threshold"]),
            "auroc": metrics.auroc(p, y),
            "ap": metrics.average_precision(p, y),
            "ece": metrics.binary_ece(p, y),
        })
    published = [l for l in labels if l["published"] and l["name"] != ip.NORMAL]
    called = np.zeros(len(truth), bool)
    for label in published:
        called |= probs[:, label["index"]] >= label["threshold"]
    has_sign = np.array([bool(names & {l["name"] for l in published}) for names in truth])
    is_normal = np.array([not names for names in truth])
    gate = next(l for l in card["labels"] if l["name"] == ip.GATE)
    quality = []
    for flag in (*ip.QUALITY_FLAGS, None):
        chosen = np.array([(flag in f) if flag else not f for f in flags], dtype=bool)
        right = called == has_sign
        quality.append({
            "flag": flag or "none",
            "images": int(chosen.sum()),
            "any_sign_agreement": float(right[chosen].mean()) if chosen.any() else math.nan,
        })
    return {
        "model_sha256": card["sha256"],
        "images": len(truth),
        "skipped": skipped,
        "labels": rows,
        "normal_with_a_sign": float(called[is_normal].mean()) if is_normal.any() else math.nan,
        "eye_gate_pass": float((probs[:, gate["index"]] >= gate["threshold"]).mean()),
        "quality": quality,
    }


def no_nan(value):
    """JSON has no NaN: write undefined metrics (e.g. a label with no images) as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: no_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [no_nan(v) for v in value]
    return value


def fmt(value):
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.2f}"


def summary(report, title=""):
    lines = [
        f"{title}Images: {report['images']} (skipped {len(report['skipped'])}), "
        f"eye gate pass {fmt(report['eye_gate_pass'])}, "
        f"Normal photos with a published sign {fmt(report['normal_with_a_sign'])}",
        "",
        "| Label | Published | n | Sensitivity (95% CI) | Specificity (95% CI) | PPV | AUROC | AP | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["labels"]:
        lo, hi = r["sensitivity_ci"]
        slo, shi = r["specificity_ci"]
        lines.append(
            f"| {r['label']} | {'yes' if r['published'] else 'no'} | {r['n_pos']} | "
            f"{fmt(r['sensitivity'])} ({fmt(lo)}-{fmt(hi)}) | {fmt(r['specificity'])} ({fmt(slo)}-{fmt(shi)}) | "
            f"{fmt(r['ppv'])} | {fmt(r['auroc'])} | {fmt(r['ap'])} | {fmt(r['ece'])} |"
        )
    lines += ["", "| Photo-quality flag | Images | Any-sign call agrees with labels |", "|---|---|---|"]
    for r in report["quality"]:
        lines.append(f"| {r['flag']} | {r['images']} | {fmt(r['any_sign_agreement'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("test_set", help="dataset labels.csv or a path,labels CSV")
    parser.add_argument("--model-dir", type=Path, help="folder with model_card.json (default model/)")
    parser.add_argument("--split", default="test", help="split of labels.csv to use, or 'all'")
    parser.add_argument("--crop", default="1.0", help="centre-crop fractions, e.g. 1.0,0.7,0.5")
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    if args.model_dir:
        ip.MODEL_DIR = args.model_dir
    session, card = ip.load_artifacts()
    items = load_items(args.test_set, args.split)
    reports = {}
    for fraction in (float(x) for x in args.crop.split(",")):
        probs, truth, skipped, flags = predict_all(session, card, items, fraction)
        if not len(truth):
            raise SystemExit("No readable images.")
        report = build_report(card, probs, truth, skipped, flags)
        reports[str(fraction)] = report
        print(summary(report, f"Centre crop {fraction:.0%}. "))
        print()
    if args.out:
        args.out.write_text(json.dumps(no_nan(reports), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
