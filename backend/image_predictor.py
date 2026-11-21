"""CPU inference for the trusted, locally supplied 12-class Keras artifact."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import hashlib
from pathlib import Path
from io import BytesIO
from threading import RLock
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.i18n import DEFAULT, t

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "model/keras_model.h5")))
LABELS_PATH = ROOT / "model/labels.txt"
MAX_BYTES = 5 * 1024 * 1024
# 25 MP covers 12-24 MP phone photos; the browser downscales larger ones before upload.
MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
EXPECTED = [
    "Normal",
    "Redness",
    "Blood Patch",
    "Swelling",
    "Discharge",
    "Watery",
    "Cornea Abnormal",
    "Eyelid Abnormal",
    "Yellow Eye",
    "Bulging Eye",
    "Pale Inner Eyelid",
    "Back Ground",
]
# Uncalibrated guards for a "lean" when the image is uncertain (top-1 below 0.60).
# Junk images (white, black, noise) measured top-1 of 0.10-0.15 with the current model.
VALID_MIN = 0.60
HINT_MIN = 0.30
HINT_MIN_MARGIN = 0.10
HINT_MAX_BACKGROUND = 0.10
# Uncalibrated photo-quality checks on the centre crop the model sees, at twice its
# resolution. They only suggest a retake; they never change the result or the triage.
QUALITY_SIZE = 448
BLUR_MIN = 60.0  # variance of the Laplacian of the grey image
DARK_MAX = 50.0  # mean grey level, 0-255
BRIGHT_MIN = 215.0
GLARE_MAX = 0.03  # share of near-white pixels (every channel at 250 or more)
QUALITY_FLAGS = ("blurry", "dark", "bright", "glare")
_lock = RLock()
_model = None
_labels = None
# SHA256 of the loaded H5, returned with each result so outputs can be traced to a file.
model_version = None


class ModelUnavailable(RuntimeError):
    pass


class InvalidImage(ValueError):
    """Rejected upload; ``key`` selects the localized message in backend.i18n."""

    def __init__(self, key):
        self.key = key
        super().__init__(t(key))


def read_labels(path=LABELS_PATH):
    labels = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(" ", 1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].strip():
            raise ValueError("labels.txt lines must look like: 10 Pale Inner Eyelid")
        labels.append((int(parts[0]), parts[1].strip()))
    if labels != list(enumerate(EXPECTED)):
        raise ValueError("labels.txt must list the 12 published classes in order.")
    return labels


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts():
    global _model, _labels, model_version
    with _lock:
        if _model is not None:
            return _model, _labels
        try:
            labels = read_labels()
            if not MODEL_PATH.is_file():
                raise FileNotFoundError("Missing model/keras_model.h5")
            import tensorflow as tf

            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            if tuple(model.input_shape) != (None, 224, 224, 3):
                raise ValueError("Model input must be (None, 224, 224, 3).")
            if tuple(model.output_shape) != (None, 12):
                raise ValueError("Model output must be (None, 12).")
            version = file_sha256(MODEL_PATH)
            _model, _labels, model_version = model, labels, version
            return _model, _labels
        except Exception as exc:
            raise ModelUnavailable(
                "Could not load the model. Check the H5 file, labels and Keras version in the server log."
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


def to_batch(image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    return ((np.asarray(image, dtype=np.float32) / 127.5) - 1.0)[None, ...]


def preprocess(image_bytes):
    return to_batch(decode(image_bytes))


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


def predict_image(image_bytes, lang=DEFAULT):
    image = decode(image_bytes)
    batch = to_batch(image)
    quality = measure_quality(image)
    quality["advice"] = [t(f"quality_{flag}", lang) for flag in quality["flags"]]
    with _lock:
        model, labels = load_artifacts()
        try:
            scores = np.asarray(model(batch, training=False), dtype=np.float32)
        except Exception as exc:
            raise ModelUnavailable("Could not run the image model.") from exc
    if scores.shape != (1, 12) or not np.isfinite(scores).all():
        raise ModelUnavailable("Invalid model output.")
    # This artifact contract is multiclass probabilities, not multilabel sigmoid or logits.
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.isclose(scores.sum(), 1, atol=0.01)
    ):
        raise ModelUnavailable("Expected a multiclass model with a 12-value softmax output.")
    index = int(np.argmax(scores[0]))
    class_id, label = labels[index]
    confidence = float(scores[0, index])
    status = (
        "invalid"
        if class_id == 11
        else ("uncertain" if confidence < VALID_MIN else "valid")
    )
    return {
        "class_id": class_id,
        "label": label,
        "confidence": confidence,
        "status": status,
        "message": t(f"image_{status}", lang),
        "hint": uncertain_hint(scores[0], labels) if status == "uncertain" else None,
        "model_version": model_version,
        "quality": quality,
    }


def uncertain_hint(scores, labels):
    """A clear lean towards one sign in an uncertain image, or None.

    The risk engine decides whether the sign matters; this only checks that the
    lead is well above the junk-image baseline and that the photo does not look
    like background.
    """
    first, second = np.argsort(scores)[::-1][:2]
    if (
        labels[first][1] in {"Normal", "Back Ground"}
        or scores[first] < HINT_MIN
        or scores[first] - scores[second] < HINT_MIN_MARGIN
        or scores[11] > HINT_MAX_BACKGROUND
    ):
        return None
    return {"label": labels[first][1], "confidence": float(scores[first])}
