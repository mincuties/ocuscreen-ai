"""A tiny ONNX model and model card that stand in for the real model in tests."""

import json
import math

import numpy as np
import pytest

import backend.image_predictor as ip

GRID = 16
# name, published, triage, threshold, hint threshold. Watery is shown but never raises
# the triage level on its own.
LABELS = [
    ("Redness", True, True, 0.5, 0.3),
    ("Blood Patch", True, True, 0.5, 0.3),
    ("Swelling", True, True, 0.5, 0.3),
    ("Discharge", True, True, 0.5, 0.3),
    ("Cataract", True, True, 0.5, None),
    ("Watery", True, False, 0.5, 0.3),
    ("Yellow Eye", False, False, 0.5, None),
    ("Normal", True, False, 0.7, None),
    ("eye_gate", False, False, 0.2, None),
]
NAMES = [name for name, *_ in LABELS]


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def build_onnx(logits, patch_logits):
    """ONNX graph whose outputs are fixed: logits [N, L] and patch_logits [N, 256, L]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n = len(logits)
    inits = [
        numpy_helper.from_array(np.zeros((1, n), np.float32), "zl"),
        numpy_helper.from_array(np.asarray(logits, np.float32)[None, :], "bl"),
        numpy_helper.from_array(np.zeros((1, 1, n), np.float32), "zp"),
        numpy_helper.from_array(np.asarray(patch_logits, np.float32)[None], "bp"),
        numpy_helper.from_array(np.array([1], np.int64), "ax1"),
        numpy_helper.from_array(np.array([1, 2], np.int64), "ax12"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["m", "ax1"], ["m2"]),
        helper.make_node("Mul", ["m2", "zl"], ["l0"]),
        helper.make_node("Add", ["l0", "bl"], ["logits"]),
        helper.make_node("Unsqueeze", ["m", "ax12"], ["m3"]),
        helper.make_node("Mul", ["m3", "zp"], ["p0"]),
        helper.make_node("Add", ["p0", "bp"], ["patch_logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fake",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 224, 224])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", n]),
            helper.make_tensor_value_info("patch_logits", TensorProto.FLOAT, ["N", GRID * GRID, n]),
        ],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def card_for(data, labels=LABELS):
    return {
        "format": 1,
        "model_file": "ocuscreen.onnx",
        "sha256": ip.hashlib.sha256(data).hexdigest(),
        "architecture": "test",
        "input": dict(ip.DEFAULT_INPUT),
        "outputs": {"logits": "logits", "patch_logits": "patch_logits", "grid": GRID},
        "labels": [
            {
                "name": name,
                "index": i,
                "published": published,
                "triage": triage,
                "platt": [1.0, 0.0],
                "threshold": threshold,
                "hint_threshold": hint,
            }
            for i, (name, published, triage, threshold, hint) in enumerate(labels)
        ],
    }


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """Install a fake model; call it with label probabilities (default 0.01, gate 0.99)."""

    def install(probs=None, patch=None, card_edit=None, onnx_bytes=None):
        probs = {"eye_gate": 0.99, **(probs or {})}
        logits = [logit(probs.get(name, 0.01)) for name in NAMES]
        patch = np.zeros((GRID * GRID, len(NAMES))) if patch is None else patch
        data = onnx_bytes or build_onnx(logits, patch)
        (tmp_path / "ocuscreen.onnx").write_bytes(data)
        card = card_for(data)
        if card_edit:
            card_edit(card)
        (tmp_path / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
        monkeypatch.setattr(ip, "MODEL_DIR", tmp_path)
        monkeypatch.setattr(ip, "_session", None)
        monkeypatch.setattr(ip, "_card", None)
        monkeypatch.setattr(ip, "model_version", None)
        return card

    return install
