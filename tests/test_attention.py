import base64
import io

import numpy as np
import pytest
from PIL import Image

import backend.image_predictor as ip
from backend import attention

tf = pytest.importorskip("tensorflow")


def tiny_model():
    """Two colour channels with opposite meanings for Redness.

    Channel 0 reacts to red, channel 1 to blue; plain grey gives zero. Redness counts
    red for and blue against, so a magenta patch (red and blue) is no evidence for it.
    A map that drops negative weights (as Grad-CAM++ does with this head) would still
    light up the magenta patch; Grad-CAM must not.
    """
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Conv2D(2, 8, strides=8, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(12, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    conv, dense = model.layers[1], model.layers[-1]
    kernel = np.zeros(conv.kernel.shape, np.float32)
    kernel[:, :, 0, 0] = 0.03  # red input -> channel 0
    kernel[:, :, 2, 1] = 0.03  # blue input -> channel 1
    # Bias -0.5 keeps each channel silent on the mid-grey background.
    conv.set_weights([kernel, np.full(2, -0.5, np.float32)])
    weights = np.zeros(dense.kernel.shape, np.float32)
    weights[:, 1] = [60.0, -60.0]  # Redness
    weights[:, 3] = [0.0, 60.0]  # Swelling: blue only
    bias = np.zeros(12, np.float32)
    bias[0] = 1.0  # Normal wins on a plain image
    dense.set_weights([weights, bias])
    return model


def png(array):
    b = io.BytesIO()
    Image.fromarray(array).save(b, format="PNG")
    return b.getvalue()


def bright_corner():
    """Mid-grey photo: red patch top left, magenta (red and blue) patch bottom right."""
    a = np.full((224, 224, 3), 128, np.uint8)
    a[:80, :80, 0] = 255
    a[-48:, -48:, 0] = 255
    a[-48:, -48:, 2] = 255
    return png(a)


@pytest.fixture
def tiny(monkeypatch):
    model = tiny_model()
    split = attention.split_model(model)
    monkeypatch.setattr(ip, "load_artifacts", lambda: (model, ip.read_labels()))
    monkeypatch.setattr(ip, "_attention", split)
    return model, split


def test_chain_matches_the_model(tiny):
    model, split = tiny
    batch = ip.preprocess(bright_corner())
    features, probs = attention.forward(split, batch)
    assert features.shape == (1, 28, 28, 2)
    assert np.allclose(probs, model(batch, training=False), atol=1e-6)


def peak(cam):
    return np.unravel_index(cam.argmax(), cam.shape)


def test_each_class_gets_its_own_map(tiny):
    _, split = tiny
    features, _ = attention.forward(split, ip.preprocess(bright_corner()))
    redness = attention.grad_cam(split[1], features, 1)
    swelling = attention.grad_cam(split[1], features, 3)
    assert redness.shape == (28, 28) and redness.max() == 1.0 and redness.min() >= 0
    assert max(peak(redness)) < 10  # the red patch
    # Blue cancels red in the magenta patch, so it is no evidence for Redness.
    assert redness[-6:, -6:].max() < 0.05
    # Swelling only looks at blue: the magenta patch, not the red one.
    assert min(peak(swelling)) > 20 and swelling[:10, :10].max() == 0


def test_predict_returns_map_only_when_asked_for_a_valid_sign(tiny):
    plain = ip.predict_image(bright_corner(), "vi")
    assert plain["label"] == "Redness" and plain["status"] == "valid"
    assert plain["attention"] is None
    shown = ip.predict_image(bright_corner(), "vi", explain=True)
    assert shown["attention"]["label"] == "Redness"
    assert shown["attention"]["note"].startswith("Vùng tô màu")
    assert {k: v for k, v in shown.items() if k != "attention"} == {
        k: v for k, v in plain.items() if k != "attention"
    }
    header, data = shown["attention"]["image"].split(",", 1)
    assert header == "data:image/jpeg;base64"
    assert Image.open(io.BytesIO(base64.b64decode(data))).size == (224, 224)
    # A black image is Normal: no map.
    black = ip.predict_image(png(np.zeros((224, 224, 3), np.uint8)), explain=True)
    assert black["label"] == "Normal" and black["attention"] is None


def test_models_without_a_softmax_head_are_refused():
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(12, activation="sigmoid")(x)
    with pytest.raises(ValueError):
        attention.split_model(tf.keras.Model(inputs, outputs))


def test_branched_models_are_refused():
    inputs = tf.keras.Input((224, 224, 3))
    a = tf.keras.layers.Conv2D(2, 8, strides=8)(inputs)
    b = tf.keras.layers.Conv2D(2, 8, strides=8)(inputs)
    x = tf.keras.layers.Concatenate()([a, b])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(12, activation="softmax")(x)
    with pytest.raises(ValueError):
        attention.split_model(tf.keras.Model(inputs, outputs))


def test_unsupported_model_keeps_predictions(monkeypatch, tmp_path):
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(12, activation="softmax")(x))
    path = tmp_path / "m.h5"
    model.save(path)
    monkeypatch.setattr(ip, "MODEL_PATH", path)
    monkeypatch.setattr(ip, "_model", None)
    monkeypatch.setattr(ip, "_attention", "stale")
    loaded, _ = ip.load_artifacts()
    assert loaded is not None and ip._attention is None
    result = ip.predict_image(bright_corner(), explain=True)
    assert result["attention"] is None
