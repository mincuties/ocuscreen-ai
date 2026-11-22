"""Shared settings for the training scripts: dataset columns, label names and paths."""

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "training" / "out"

# Dataset column -> image sign name used by the backend and data/disease_knowledge.json.
# Columns with fewer than MIN_POSITIVES positives are not trained (cornea_abnormal,
# watery and eyelid_abnormal in _clean_v1).
SIGNS = {
    "redness": "Redness",
    "blood_patch": "Blood Patch",
    "swelling": "Swelling",
    "discharge": "Discharge",
    "watery": "Watery",
    "cornea_abnormal": "Cornea Abnormal",
    "eyelid_abnormal": "Eyelid Abnormal",
    "yellow_eye": "Yellow Eye",
    "bulging_eye": "Bulging Eye",
    "pale_inner_eyelid": "Pale Inner Eyelid",
    "cataract": "Cataract",
}
NORMAL = "normal"
GATE = "eye_gate"
MIN_POSITIVES = 10

# Same preprocessing contract as the server (written into model_card.json).
INPUT_SPEC = {
    "name": "image",
    "size": 224,
    "layout": "NCHW",
    "resize": "center_crop",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
GRID = INPUT_SPEC["size"] // 14


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(data_dir):
    """Rows of labels.csv with 0/1 columns as ints."""
    path = Path(data_dir) / "labels.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    missing = (set(SIGNS) | {NORMAL, "split", "dup_group", "source", "file"}) - rows[0].keys()
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    for row in rows:
        for column in (*SIGNS, NORMAL):
            row[column] = int(row[column])
    return rows


def trained_columns(rows):
    """Sign columns with enough positives in train+val, then normal."""
    fit = [r for r in rows if r["split"] != "test"]
    columns = [c for c in SIGNS if sum(r[c] for r in fit) >= MIN_POSITIVES]
    return columns + [NORMAL]


def output_names(columns):
    """Names of the head's outputs in order: trained signs, Normal, then the eye gate."""
    return [SIGNS.get(c, "Normal") for c in columns] + [GATE]
