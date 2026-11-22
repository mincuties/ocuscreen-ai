"""CPU inference for the locally supplied ONNX model described by model/model_card.json."""

import hashlib
import json
import logging
import math
import os
from io import BytesIO
from pathlib import Path
from threading import RLock
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend import attention
from backend.i18n import DEFAULT, localize, t
from backend.risk_engine import KB

log = logging.getLogger("ocuscreen")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "model")))
CARD_NAME = "model_card.json"
MAX_BYTES = 5 * 1024 * 1024
# 25 MP covers 12-24 MP phone photos; the browser downscales larger ones before upload.
MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
NORMAL = "Normal"
GATE = "eye_gate"
# Preprocessing used when no model card is loaded (the training default).
DEFAULT_INPUT = {
    "name": "image",
    "size": 224,
    "layout": "NCHW",
    "resize": "center_crop",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
RESIZE_MODES = ("center_crop", "letterbox")
# A borderline sign only becomes the hint of an uncertain photo when it leads the
# next borderline sign by this much probability.
HINT_MARGIN = 0.10
# Uncalibrated photo-quality checks on the centre crop the model sees, at twice its
# resolution. They only suggest a retake; they never change the result or the triage.
QUALITY_SIZE = 448
BLUR_MIN = 60.0  # variance of the Laplacian of the grey image
DARK_MAX = 50.0  # mean grey level, 0-255
BRIGHT_MIN = 215.0
GLARE_MAX = 0.03  # share of near-white pixels (every channel at 250 or more)
QUALITY_FLAGS = ("blurry", "dark", "bright", "glare")
ORT_THREADS = int(os.getenv("ORT_NUM_THREADS", "2"))
_lock = RLock()
_session = None
_card = None
# SHA256 of the loaded ONNX file, returned with each result so outputs can be traced.
model_version = None


class ModelUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    """Rejected upload; ``key`` selects the localized message in backend.i18n."""

    def __init__(self, key):
        self.key = key
        super().__init__(t(key))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_card(card):
    """Return the card or raise ValueError on anything that would silently misbehave."""
    spec = card.get("input", {})
    problems = [
        card.get("format") != 1,
        not isinstance(card.get("model_file"), str) or "/" in card["model_file"] or "\\" in card["model_file"],
        not isinstance(card.get("sha256"), str) or len(card["sha256"]) != 64,
        not isinstance(spec.get("name"), str),
        not isinstance(spec.get("size"), int) or spec["size"] < 32,
        spec.get("layout") != "NCHW",
        spec.get("resize") not in RESIZE_MODES,
        not all(isinstance(spec.get(k), list) and len(spec[k]) == 3 and all(map(_number, spec[k])) for k in ("mean", "std")),
        any(s <= 0 for s in spec.get("std") or [0]),
        not isinstance(card.get("outputs", {}).get("grid"), int),
        not isinstance(card.get("demo", False), bool),
    ]
    if any(problems):
        raise ValueError("model_card.json: invalid header or input section")
    labels = card.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("model_card.json: labels must be a non-empty list")
    names = [label.get("name") for label in labels]
    if len(set(names)) != len(names) or names.count(GATE) != 1 or NORMAL not in names:
        raise ValueError("model_card.json: label names must be unique and include Normal and eye_gate")
    known = set(KB["image_signs"])
    for i, label in enumerate(labels):
        platt = label.get("platt")
        hint = label.get("hint_threshold")
        bad = [
            label.get("index") != i,
            label["name"] != GATE and label["name"] not in known,
            not isinstance(label.get("published"), bool),
            # Optional; a card without it lets no label raise the triage level alone.
            not isinstance(label.get("triage", False), bool),
            label.get("triage", False) and not label["published"],
            label["name"] == GATE and label["published"],
            not (isinstance(platt, list) and len(platt) == 2 and all(map(_number, platt))),
            not _number(label.get("threshold")) or not 0 < label["threshold"] < 1,
            hint is not None and (not _number(hint) or not 0 < hint < label["threshold"]),
        ]
        if any(bad):
            raise ValueError(f"model_card.json: invalid label {label.get('name')!r}")
    return card


def read_card(path=None):
    path = Path(path) if path else MODEL_DIR / CARD_NAME
    return check_card(json.loads(path.read_text(encoding="utf-8")))


def check_session(session, card):
    """The ONNX graph must match the card: input name and shape, output names and sizes."""
    spec, outputs = card["input"], card["outputs"]
    n, grid = len(card["labels"]), card["outputs"]["grid"]
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != spec["name"] or list(inputs[0].shape[1:]) != [3, spec["size"], spec["size"]]:
        raise ValueError("ONNX input does not match model_card.json")
    shapes = {o.name: list(o.shape) for o in session.get_outputs()}
    logits, patches = shapes.get(outputs.get("logits")), shapes.get(outputs.get("patch_logits"))
    if logits is None or logits[1:] != [n]:
        raise ValueError("ONNX logits do not match the labels in model_card.json")
    if patches is None or patches[1:] != [grid * grid, n]:
        raise ValueError("ONNX patch_logits do not match model_card.json")


def load_artifacts():
    global _session, _card, model_version
    with _lock:
        if _session is not None:
            return _session, _card
        try:
            card = read_card()
            path = MODEL_DIR / card["model_file"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing model/{card['model_file']}")
            version = file_sha256(path)
            if version != card["sha256"]:
                raise ValueError("The ONNX file does not match the sha256 in model_card.json")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = ORT_THREADS
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            check_session(session, card)
            _session, _card, model_version = session, card, version
            return _session, _card
        except Exception as exc:
            raise ModelUnavailable(
                "Could not load the model. Check model/model_card.json and the ONNX file in the server log."
            ) from exc


def decode(image_bytes):
    """Validate untrusted bytes and return an upright RGB image."""
    if not image_bytes or len(image_bytes) > MAX_BYTES:
        raise InvalidImage("image_size")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Only let the three accepted decoders parse untrusted bytes; without this
            # Pillow probes every registered format plugin before the check below.
            with Image.open(
                BytesIO(image_bytes), formats=("JPEG", "PNG", "WEBP")
            ) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InvalidImage("image_format")
                if getattr(original, "n_frames", 1) != 1:
                    raise InvalidImage("image_animated")
                if (
                    min(original.size) < 64
                    or original.width * original.height > MAX_PIXELS
                ):
                    raise InvalidImage("image_dimensions")
                return ImageOps.exif_transpose(original).convert("RGB")
    except InvalidImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        # Pillow refuses oversized images before the size check above can run.
        raise InvalidImage("image_dimensions") from exc
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise InvalidImage("image_unreadable") from exc


def model_view(image, spec):
    """The square image the model sees (also the background of the attention map)."""
    size = (spec["size"], spec["size"])
    if spec["resize"] == "letterbox":
        fill = tuple(int(round(m * 255)) for m in spec["mean"])
        return ImageOps.pad(image, size, Image.Resampling.LANCZOS, color=fill)
    return ImageOps.fit(image, size, Image.Resampling.LANCZOS)


def to_batch(image, spec):
    """(1, 3, size, size) float32, scaled to [0, 1] then normalized with the card's mean/std.

    The training scripts import this function, so train and serve share one path.
    """
    a = np.asarray(model_view(image, spec), dtype=np.float32) / 255.0
    a = (a - np.asarray(spec["mean"], dtype=np.float32)) / np.asarray(spec["std"], dtype=np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None, ...])


def preprocess(image_bytes, spec=DEFAULT_INPUT):
    return to_batch(decode(image_bytes), spec)


def measure_quality(image):
    """Sharpness, brightness and glare of the centre crop, plus the flags they raise."""
    crop = ImageOps.fit(image, (QUALITY_SIZE,) * 2, Image.Resampling.LANCZOS)
    grey = np.asarray(crop.convert("L"), dtype=np.float32)
    laplacian = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
        - 4 * grey[1:-1, 1:-1]
    )
    sharpness = float(laplacian.var())
    brightness = float(grey.mean())
    glare = float((np.asarray(crop).min(axis=2) >= 250).mean())
    raised = {
        "blurry": sharpness < BLUR_MIN,
        "dark": brightness < DARK_MAX,
        "bright": brightness > BRIGHT_MIN,
        "glare": glare > GLARE_MAX,
    }
    return {
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "glare": round(glare, 4),
        "flags": [flag for flag in QUALITY_FLAGS if raised[flag]],
    }


def run_model(session, card, batch):
    outputs = card["outputs"]
    try:
        logits, patches = session.run(
            [outputs["logits"], outputs["patch_logits"]], {card["input"]["name"]: batch}
        )
    except Exception as exc:
        raise ModelUnavailable("Could not run the image model.") from exc
    n, grid = len(card["labels"]), outputs["grid"]
    logits, patches = np.asarray(logits, np.float64), np.asarray(patches, np.float64)
    if (
        logits.shape != (1, n)
        or patches.shape != (1, grid * grid, n)
        or not np.isfinite(logits).all()
        or not np.isfinite(patches).all()
    ):
        raise ModelUnavailable("Invalid model output.")
    return logits[0], patches[0]


def calibrated(card, logits):
    """Per-label probability after the card's Platt scaling (independent sigmoids)."""
    a = np.array([label["platt"][0] for label in card["labels"]])
    b = np.array([label["platt"][1] for label in card["labels"]])
    return 1 / (1 + np.exp(-(a * logits + b)))


def interpret(card, probs):
    """Status, findings and the fields the risk engine and the page read."""
    labels = card["labels"]
    by_name = {label["name"]: label for label in labels}
    gate = by_name[GATE]
    published = [l for l in labels if l["published"] and l["name"] not in (NORMAL, GATE)]
    findings = []
    for label in published:
        p = float(probs[label["index"]])
        if p >= label["threshold"]:
            state = "positive"
        elif label["hint_threshold"] is not None and p >= label["hint_threshold"]:
            state = "borderline"
        else:
            continue
        findings.append(
            {
                "label": label["name"],
                "probability": p,
                "threshold": label["threshold"],
                "state": state,
                "triage": label.get("triage", False),
            }
        )
    findings.sort(key=lambda f: (f["state"] != "positive", -f["probability"] / f["threshold"]))
    positives = [f for f in findings if f["state"] == "positive"]
    normal = by_name[NORMAL]
    normal_p = float(probs[normal["index"]])
    shown = {l["name"] for l in published} | ({NORMAL} if normal["published"] else set())
    # Signs the knowledge base knows but this model does not report: the page must not
    # read their absence as "no sign".
    unassessed = [s for s in KB["image_signs"] if s != NORMAL and s not in shown]
    if float(probs[gate["index"]]) < gate["threshold"]:
        status, label, confidence = "invalid", None, None
    elif positives:
        status, label, confidence = "valid", positives[0]["label"], positives[0]["probability"]
    elif normal["published"] and normal_p >= normal["threshold"]:
        status, label, confidence = "valid", NORMAL, normal_p
    else:
        status, label, confidence = "uncertain", None, None
    hint = None
    if status == "uncertain":
        borderline = sorted((f for f in findings if f["state"] == "borderline"), key=lambda f: -f["probability"])
        if borderline and (len(borderline) == 1 or borderline[0]["probability"] - borderline[1]["probability"] >= HINT_MARGIN):
            top = borderline[0]
            hint = {"label": top["label"], "confidence": top["probability"], "triage": top["triage"]}
    ranked = sorted(
        (l for l in labels if l["name"] in shown), key=lambda l: -probs[l["index"]]
    )
    top3 = [{"label": l["name"], "confidence": float(probs[l["index"]])} for l in ranked[:3]]
    return {
        "status": status,
        "label": label,
        "confidence": confidence,
        "findings": findings if status != "invalid" else [],
        "unassessed": unassessed,
        "hint": hint,
        "top3": top3 if status != "invalid" else None,
    }


def predict_image(image_bytes, lang=DEFAULT, explain=False):
    image = decode(image_bytes)
    quality = measure_quality(image)
    quality["advice"] = [t(f"quality_{flag}", lang) for flag in quality["flags"]]
    with _lock:
        session, card = load_artifacts()
        batch = to_batch(image, card["input"])
        logits, patches = run_model(session, card, batch)
    probs = calibrated(card, logits)
    result = interpret(card, probs)
    attention_map = None
    if explain and result["status"] == "valid" and result["label"] != NORMAL:
        index = next(l["index"] for l in card["labels"] if l["name"] == result["label"])
        try:
            cam = attention.patch_map(patches[:, index], card["outputs"]["grid"])
            if cam is not None:
                attention_map = {
                    "label": result["label"],
                    "image": attention.overlay(model_view(image, card["input"]), cam),
                    "note": t("attention_note", lang),
                }
        except Exception:
            # The map is optional; the prediction stands without it.
            log.exception("Could not compute the attention map")
    message = t(f"image_{result['status']}", lang)
    note = None
    if result["unassessed"] and result["status"] != "invalid":
        signs = ", ".join(localize(KB["image_signs"][s]["description"], lang) for s in result["unassessed"])
        note = t("image_unassessed", lang).format(signs=signs)
    return {
        "class_id": None,
        **result,
        "message": message,
        "unassessed_note": note,
        "attention": attention_map,
        "model_version": model_version,
        "demo": card.get("demo", False),
        "quality": quality,
    }
