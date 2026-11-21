"""CPU inference for the trusted, locally supplied 12-class Keras artifact."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
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
Image.MAX_IMAGE_PIXELS = 12_000_000
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
_lock = RLock()
_model = None
_labels = None


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


def load_artifacts():
    global _model, _labels
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
            _model, _labels = model, labels
            return _model, _labels
        except Exception as exc:
            raise ModelUnavailable(
                "Could not load the model. Check the H5 file, labels and Keras version in the server log."
            ) from exc


def preprocess(image_bytes):
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
                    or original.width * original.height > 12_000_000
                ):
                    raise InvalidImage("image_dimensions")
                image = ImageOps.exif_transpose(original).convert("RGB")
                image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                return ((np.asarray(image, dtype=np.float32) / 127.5) - 1.0)[None, ...]
    except InvalidImage:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise InvalidImage("image_unreadable") from exc


def predict_image(image_bytes, lang=DEFAULT):
    batch = preprocess(image_bytes)
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
        "invalid" if class_id == 11 else ("uncertain" if confidence < 0.60 else "valid")
    )
    return {
        "class_id": class_id,
        "label": label,
        "confidence": confidence,
        "status": status,
        "message": t(f"image_{status}", lang),
    }
