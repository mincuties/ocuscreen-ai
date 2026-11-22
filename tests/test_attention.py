import numpy as np
import pytest
from PIL import Image

from backend import attention
import backend.image_predictor as ip
from conftest import GRID, NAMES
from test_app import photo


def test_patch_map_shape_peak_and_negative_cases():
    c = np.zeros(GRID * GRID)
    c[5] = 2.0
    c[6] = 1.0
    c[7] = -0.5
    cam = attention.patch_map(c, GRID)
    assert cam.shape == (GRID, GRID) and cam.max() == 1.0
    assert cam.flat[6] == pytest.approx(0.5) and cam.flat[7] == 0
    # Patches that together lower the score give no map.
    assert attention.patch_map(-np.abs(c), GRID) is None
    assert attention.patch_map(np.zeros(GRID * GRID), GRID) is None
    assert attention.patch_map(np.ones(10), GRID) is None


def test_overlay_is_a_jpeg_of_the_photo_size():
    cam = np.zeros((GRID, GRID))
    cam[0, 0] = 1
    url = attention.overlay(Image.new("RGB", (224, 224), "gray"), cam)
    assert url.startswith("data:image/jpeg;base64,")


def _patch_for(label, cell):
    patch = np.zeros((GRID * GRID, len(NAMES)))
    patch[cell, NAMES.index(label)] = 1.0
    return patch


def test_map_follows_the_chosen_label(fake_model):
    patch = _patch_for("Redness", 3) + _patch_for("Discharge", 200)
    fake_model({"Redness": 0.9, "Discharge": 0.6}, patch=patch)
    result = ip.predict_image(photo(), explain=True)
    assert result["attention"]["label"] == "Redness"
    assert result["attention"]["image"].startswith("data:image/jpeg;base64,")


def test_no_map_unless_asked_for_a_valid_sign(fake_model):
    patch = _patch_for("Redness", 3) + _patch_for("Normal", 3)
    fake_model({"Redness": 0.9}, patch=patch)
    assert ip.predict_image(photo())["attention"] is None
    fake_model({"Normal": 0.95}, patch=patch)
    assert ip.predict_image(photo(), explain=True)["attention"] is None
    fake_model({"Redness": 0.4}, patch=patch)  # uncertain
    assert ip.predict_image(photo(), explain=True)["attention"] is None


def test_no_map_when_patches_lower_the_score(fake_model):
    fake_model({"Redness": 0.9}, patch=-_patch_for("Redness", 3))
    result = ip.predict_image(photo(), explain=True)
    assert result["status"] == "valid" and result["attention"] is None
