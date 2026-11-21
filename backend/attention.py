"""Grad-CAM map of where the image model looked, for a valid, non-Normal result.

The map shows which part of the photo drove the model's score for the chosen label.
It is not the location of a lesion and is never used by the triage.

Grad-CAM, not Grad-CAM++: with a GlobalAveragePooling -> Dense head the gradient is
the same at every position, so Grad-CAM++ keeps only the channels with a positive
Dense weight and gives almost the same map for every class (measured on the current
model: maps of different classes correlated at 1.0). Grad-CAM keeps the sign of every
weight, which for this head is exactly the class activation map.
"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image

OVERLAY_ALPHA = 0.6


def split_model(model):
    """(body layers up to the last 4D feature map, head layers) or ValueError.

    The model must be a plain chain of layers (as the current Rescaling ->
    EfficientNetV2-S -> GlobalAveragePooling -> Dropout -> Dense model is) ending in a
    softmax Dense layer. The chain is checked against the model's own output.
    """
    import tensorflow as tf

    layers = [
        layer
        for layer in model.layers
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    feature_index = max(
        (i for i, layer in enumerate(layers) if len(layer.output.shape) == 4),
        default=None,
    )
    last = layers[-1] if layers else None
    if (
        feature_index is None
        or not isinstance(last, tf.keras.layers.Dense)
        or last.activation is not tf.keras.activations.softmax
    ):
        raise ValueError("Attention maps need a 4D feature map and a softmax Dense head.")
    split = layers[: feature_index + 1], layers[feature_index + 1 :]
    probe = np.random.default_rng(0).uniform(-1, 1, (1, *model.input_shape[1:]))
    probe = probe.astype(np.float32)
    _, chained = forward(split, probe)
    if not np.allclose(chained, model(probe, training=False), atol=1e-5):
        raise ValueError("The model is not a plain chain of layers.")
    return split


def forward(split, batch):
    """(feature map, class probabilities) in one pass through the chain."""
    x = batch
    for layer in split[0]:
        x = layer(x, training=False)
    features, y = x, x
    for layer in split[1]:
        y = layer(y, training=False)
    return features, y


def logits(head, features):
    """Pre-softmax scores: softmax saturates, which flattens the gradients."""
    import tensorflow as tf

    x = features
    for layer in head[:-1]:
        x = layer(x, training=False)
    return tf.matmul(x, head[-1].kernel) + head[-1].bias


def grad_cam(head, features, class_index):
    """Map in [0, 1] with the feature map's height and width.

    Scaled to its own peak, so colours compare areas within one photo, not photos.
    """
    import tensorflow as tf

    features = tf.convert_to_tensor(features)
    with tf.GradientTape() as tape:
        tape.watch(features)
        score = logits(head, features)[0, class_index]
    grads = tape.gradient(score, features)[0].numpy().astype(np.float64)
    maps = features[0].numpy().astype(np.float64)
    weights = grads.mean(axis=(0, 1))
    cam = np.maximum((maps * weights).sum(axis=2), 0)
    peak = cam.max()
    return cam / peak if peak > 0 else cam


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
