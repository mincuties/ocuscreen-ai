"""Compute frozen DINOv2 embeddings for every image and augmented view.

    python training/extract.py --data "C:/path/_clean_v1" --views 16

Writes training/out/features.npz. Each row is one view of one image: the
concatenation of the [CLS] token and the mean of the patch tokens (768 values for
ViT-S/14). View 0 goes through exactly the server preprocessing; views 1..N-1 are
random crops (scale 0.5-1), flips, rescaling, brightness/contrast, blur and JPEG
re-compression. Hue, saturation and white balance are never changed: redness,
yellow and pale are colour signs.

Test images also get centre zooms of 70% and 50% for the robustness report.
Non-eye images for the eye gate come from excluded.csv (`not_usable_photo`) plus
generated blank, gradient and noise images.
"""

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from common import INPUT_SPEC, OUT, read_rows, sha256
from backend.image_predictor import to_batch

HUB_REPO = "facebookresearch/dinov2"
HUB_MODEL = "dinov2_vits14_reg"
ZOOMS = (0.7, 0.5)
SYNTHETIC_NEGATIVES = 120


def trim_dark_border(image, level=12, min_keep=0.6):
    """Crop rows/columns that are almost black at the edges (rotation corners, bars).

    The black borders are source fingerprints in this dataset; nothing is cut if the
    result would keep less than ``min_keep`` of either side.
    """
    grey = np.asarray(image.convert("L"))
    rows = np.where(grey.max(axis=1) > level)[0]
    cols = np.where(grey.max(axis=0) > level)[0]
    if not len(rows) or not len(cols):
        return image
    box = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
    w, h = box[2] - box[0], box[3] - box[1]
    if w < min_keep * image.width or h < min_keep * image.height:
        return image
    return image.crop(box)


def random_view(image, rng):
    """One augmented copy of a PIL image (no colour changes)."""
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    # Rescale so that the original resolution is a weaker fingerprint.
    factor = rng.uniform(0.5, 1.0)
    if factor < 0.98 and min(image.size) * factor >= 112:
        size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        image = image.resize(size, Image.Resampling.BILINEAR)
    # Random crop of 50-100% of the area, aspect ratio 0.8-1.25.
    area = image.width * image.height
    for _ in range(10):
        target = area * rng.uniform(0.5, 1.0)
        ratio = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
        w = int(round((target * ratio) ** 0.5))
        h = int(round((target / ratio) ** 0.5))
        if 0 < w <= image.width and 0 < h <= image.height:
            x = rng.randint(0, image.width - w)
            y = rng.randint(0, image.height - h)
            image = image.crop((x, y, x + w, y + h))
            break
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(60, 95))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def centre_zoom(image, fraction):
    w, h = image.size
    side = int(min(w, h) * fraction)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def synthetic_negatives(n, seed=0):
    """Blank, gradient, noise and stripe images, a floor for the eye gate."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            a = np.full((256, 256, 3), rng.integers(0, 256, 3), dtype=np.uint8)
        elif kind == 1:
            t = np.linspace(0, 1, 256)[None, :, None]
            c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
            a = np.broadcast_to(c0 + (c1 - c0) * t, (256, 256, 3)).astype(np.uint8)
        elif kind == 2:
            a = rng.normal(128, rng.uniform(10, 80), (256, 256, 3)).clip(0, 255).astype(np.uint8)
        else:
            period = int(rng.integers(4, 40))
            stripes = ((np.arange(256) // period) % 2 * 255).astype(np.uint8)
            a = np.repeat(np.repeat(stripes[None, :, None], 256, 0), 3, 2)
        images.append(Image.fromarray(a))
    return images


def load_backbone():
    import torch

    torch.set_grad_enabled(False)
    model = torch.hub.load(HUB_REPO, HUB_MODEL)
    model.eval()
    hub_dir = Path(torch.hub.get_dir())
    weights = sorted((hub_dir / "checkpoints").glob("dinov2_vits14_reg4*.pth"))
    return model, {
        "hub_repo": HUB_REPO,
        "hub_model": HUB_MODEL,
        "weights_file": weights[0].name if weights else None,
        "weights_sha256": sha256(weights[0]) if weights else None,
        "torch": torch.__version__,
    }


def embed(model, batches):
    """(N, 768) float32 features for an (N, 3, H, W) normalized array."""
    import torch

    x = torch.from_numpy(np.concatenate(batches)).float()
    out = model.forward_features(x)
    cls = out["x_norm_clstoken"]
    patches = out["x_norm_patchtokens"]
    return torch.cat([cls, patches.mean(dim=1)], dim=1).numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="folder with labels.csv")
    parser.add_argument("--views", type=int, default=16, help="views per image, view 0 included")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    args = parser.parse_args()

    rows = read_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    raw_root = args.data.parent
    negatives = []
    excluded = args.data / "excluded.csv"
    if excluded.is_file():
        with open(excluded, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("reason") == "not_usable_photo":
                    path = raw_root / row["orig_path"]
                    if path.is_file():
                        negatives.append(path)
    if args.limit:
        negatives = negatives[: max(4, args.limit // 4)]

    model, provenance = load_backbone()
    rng = random.Random(args.seed)
    feats, meta = [], []  # meta: (kind, index, view)
    pending, pending_meta = [], []

    def flush():
        if pending:
            feats.append(embed(model, pending))
            meta.extend(pending_meta)
            pending.clear()
            pending_meta.clear()

    def add(image, kind, index, view):
        pending.append(to_batch(image, INPUT_SPEC))
        pending_meta.append((kind, index, view))
        if len(pending) == args.batch:
            flush()

    start = time.time()
    skipped = []
    for i, row in enumerate(rows):
        try:
            with Image.open(args.data / row["file"]) as im:
                image = trim_dark_border(ImageOps.exif_transpose(im).convert("RGB"))
        except OSError as exc:
            skipped.append({"file": row["file"], "reason": str(exc)})
            continue
        add(image, "eye", i, 0)
        for v in range(1, args.views):
            add(random_view(image, rng), "eye", i, v)
        if row["split"] == "test":
            for z, fraction in enumerate(ZOOMS):
                add(centre_zoom(image, fraction), "zoom", i, z)
        if i % 100 == 0:
            print(f"{i}/{len(rows)} images, {time.time() - start:.0f}s", flush=True)

    neg_images = []
    for path in negatives:
        try:
            with Image.open(path) as im:
                neg_images.append(ImageOps.exif_transpose(im).convert("RGB"))
        except OSError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
    neg_images += synthetic_negatives(SYNTHETIC_NEGATIVES, args.seed)
    for j, image in enumerate(neg_images):
        add(image, "neg", j, 0)
        for v in range(1, 4):
            add(random_view(image, rng), "neg", j, v)
    flush()

    X = np.concatenate(feats).astype(np.float16)
    kinds = np.array([m[0] for m in meta])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "features.npz",
        X=X,
        kind=kinds,
        index=np.array([m[1] for m in meta], dtype=np.int32),
        view=np.array([m[2] for m in meta], dtype=np.int16),
        image_id=np.array([r["image_id"] for r in rows]),
    )
    info = {
        **provenance,
        "labels_csv_sha256": sha256(args.data / "labels.csv"),
        "images": len(rows),
        "views": args.views,
        "negatives_real": len(negatives),
        "negatives_synthetic": SYNTHETIC_NEGATIVES,
        "zooms": list(ZOOMS),
        "skipped": skipped,
        "seconds": round(time.time() - start),
        "limit": args.limit,
    }
    (OUT / "features.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Saved {X.shape[0]} rows of {X.shape[1]} features in {info['seconds']}s")


if __name__ == "__main__":
    main()
