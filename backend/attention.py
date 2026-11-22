"""Map of which image patches pushed the score of the chosen label up.

The model's head is linear on [CLS token, mean of patch tokens], so the mean-patch
part of a label's logit splits exactly into one number per patch; the ONNX model
returns them as `patch_logits` (see training/export.py). The map shows the positive
part of those numbers on the model's patch grid.

Limits: it explains only the mean-patch part of the logit, not the CLS part, and each
patch token already mixes information from the whole photo. It is not the location of
a lesion and is never used by the triage.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

OVERLAY_ALPHA = 0.6


def patch_map(contributions, grid):
    """Map in [0, 1] of shape (grid, grid), or None when patches did not support the label.

    Scaled to its own peak, so colours compare areas within one photo, not photos. When
    the patches together pull the label's score down, there is nothing honest to show.
    """
    c = np.asarray(contributions, dtype=np.float64)
    if c.shape != (grid * grid,) or not np.isfinite(c).all() or c.sum() <= 0:
        return None
    cam = np.maximum(c, 0).reshape(grid, grid)
    return cam / cam.max()


def colorize(cam):
    """Blue (low) to red (high) RGB array for values in [0, 1]."""
    cam = np.clip(cam, 0, 1)[..., None]
    red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.concatenate([red, green, blue], axis=-1) * 255).astype(np.uint8)


def overlay(image, cam):
    """JPEG data URL of the map over the crop the model saw."""
    size = image.size
    resized = Image.fromarray(cam.astype(np.float32)).resize(
        size, Image.Resampling.BILINEAR
    )
    cam = np.clip(np.asarray(resized), 0, 1)
    # Weak areas keep the photo as it is; strong areas take more of the colour.
    alpha = (OVERLAY_ALPHA * cam)[..., None]
    photo = np.asarray(image.convert("RGB"), dtype=np.float32)
    mixed = photo * (1 - alpha) + colorize(cam).astype(np.float32) * alpha
    blended = Image.fromarray(mixed.round().astype(np.uint8))
    buffer = BytesIO()
    blended.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
