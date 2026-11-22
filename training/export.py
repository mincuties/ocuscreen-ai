"""Export DINOv2 + the fitted head to ONNX and write model/model_card.json.

    python training/export.py --data "C:/path/_clean_v1"

Outputs, all in model/:
- ocuscreen.onnx: input `image` float32 [N, 3, 224, 224] (NCHW, ImageNet-normalized);
  outputs `logits` [N, L] and `patch_logits` [N, 256, L].
- model_card.json: preprocessing, labels, calibration, thresholds and provenance.
- preprocess_golden.npz: one test photo, its input tensor and expected logits, used
  by the backend tests to catch any train/serve preprocessing difference.

`patch_logits[:, i, j]` is the share of label j's logit that comes through patch i in
the mean-patch half of the features, centred on the training mean, so that
sum_i patch_logits[:, i, j] = W_patch[:, j] . (mean_patch - patch_mean).
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import GRID, INPUT_SPEC, OUT, ROOT, read_rows, sha256
from extract import load_backbone
from backend.image_predictor import to_batch

MODEL_DIR = ROOT / "model"
ONNX_NAME = "ocuscreen.onnx"


def build(backbone, head):
    import torch

    class Exported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            W = torch.from_numpy(head["W"]).float()
            half = W.shape[0] // 2
            self.register_buffer("W", W)
            self.register_buffer("b", torch.from_numpy(head["b"]).float())
            self.register_buffer("W_patch", W[half:] / (GRID * GRID))
            self.register_buffer("patch_mean", torch.from_numpy(head["patch_mean"]).float())
            # The input size is fixed, so compute the interpolated position embedding once
            # in torch: its antialiased bicubic resize has no ONNX operator.
            size = INPUT_SPEC["size"]
            with torch.no_grad():
                tokens = torch.zeros(1, GRID * GRID + 1, backbone.embed_dim)
                pos = backbone.interpolate_pos_encoding(tokens, size, size)
            self.register_buffer("pos_embed_fixed", pos.clone())
            backbone.interpolate_pos_encoding = lambda x, w, h: self.pos_embed_fixed

        def forward(self, image):
            out = self.backbone.forward_features(image)
            cls = out["x_norm_clstoken"]
            patches = out["x_norm_patchtokens"]
            features = torch.cat([cls, patches.mean(dim=1)], dim=1)
            logits = features @ self.W + self.b
            patch_logits = (patches - self.patch_mean) @ self.W_patch
            return logits, patch_logits

    return Exported().eval()


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="folder with labels.csv")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch
    import onnxruntime as ort

    head = dict(np.load(OUT / "head.npz"))
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    features_info = json.loads((OUT / "features.json").read_text(encoding="utf-8"))
    backbone, provenance = load_backbone()
    model = build(backbone, head)

    rows = read_rows(args.data)
    test_rows = [r for r in rows if r["split"] == "test"][:20]
    images = []
    for r in test_rows:
        with Image.open(args.data / r["file"]) as im:
            images.append(ImageOps.exif_transpose(im).convert("RGB"))
    batch = np.concatenate([to_batch(im, INPUT_SPEC) for im in images]).astype(np.float32)

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / ONNX_NAME
    example = torch.from_numpy(batch[:1])
    export_kwargs = dict(
        input_names=["image"],
        output_names=["logits", "patch_logits"],
        dynamic_axes={"image": {0: "N"}, "logits": {0: "N"}, "patch_logits": {0: "N"}},
        opset_version=args.opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(model, (example,), str(path), dynamo=False, **export_kwargs)
        except TypeError:  # torch without the dynamo argument
            torch.onnx.export(model, (example,), str(path), **export_kwargs)

    with torch.no_grad():
        ref_logits, ref_patch = (t.numpy() for t in model(torch.from_numpy(batch)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_logits, onnx_patch = session.run(None, {"image": batch})
    diff = float(max(np.abs(onnx_logits - ref_logits).max(), np.abs(onnx_patch - ref_patch).max()))
    print(f"max |onnx - torch| = {diff:.2e} on {len(batch)} test photos")
    if diff > 1e-3:
        raise SystemExit("ONNX output differs from PyTorch; not writing the model card.")

    first = test_rows[0]
    with open(args.data / first["file"], "rb") as f:
        golden_bytes = f.read()
    np.savez_compressed(
        MODEL_DIR / "preprocess_golden.npz",
        image_bytes=np.frombuffer(golden_bytes, dtype=np.uint8),
        tensor=batch[:1],
        logits=onnx_logits[:1],
    )

    card = {
        "format": 1,
        "model_file": ONNX_NAME,
        "sha256": sha256(path),
        "architecture": f"{provenance['hub_model']} (frozen) + linear head",
        "input": INPUT_SPEC,
        "outputs": {"logits": "logits", "patch_logits": "patch_logits", "grid": GRID},
        "labels": report["card_labels"],
        "training": {
            "labels_csv_sha256": features_info["labels_csv_sha256"],
            "weights_file": provenance["weights_file"],
            "weights_sha256": provenance["weights_sha256"],
            "torch": provenance["torch"],
            "git_commit": git_commit(),
            "date": date.today().isoformat(),
            "onnx_max_abs_diff": diff,
        },
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and model_card.json")


if __name__ == "__main__":
    main()
