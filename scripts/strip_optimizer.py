"""Re-save the Keras H5 model without its optimizer state and check the outputs match.

The optimizer state is only needed to resume training; dropping it shrinks the file
(the current model goes from about 198 MB to about 82 MB) without changing predictions.

    python scripts/strip_optimizer.py model/keras_model.h5 model/keras_model_slim.h5
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.target.resolve() == args.source.resolve():
        sys.exit("Write to a new file; do not overwrite the source model.")

    import tensorflow as tf

    original = tf.keras.models.load_model(args.source, compile=False)
    original.save(args.target, include_optimizer=False)
    slim = tf.keras.models.load_model(args.target, compile=False)

    # Fixed inputs in the model's [-1, 1] range: random noise plus flat images.
    rng = np.random.default_rng(0)
    batch = rng.uniform(-1, 1, (args.samples, 224, 224, 3)).astype(np.float32)
    flat = [np.full((1, 224, 224, 3), v, np.float32) for v in (-1.0, 0.0, 1.0)]
    batch = np.concatenate([batch, *flat])
    before = original(batch, training=False).numpy()
    after = slim(batch, training=False).numpy()
    diff = float(np.abs(before - after).max())
    print(f"source {args.source.stat().st_size / 1e6:.1f} MB sha256 {sha256(args.source)}")
    print(f"target {args.target.stat().st_size / 1e6:.1f} MB sha256 {sha256(args.target)}")
    print(f"max output difference on {len(batch)} fixed inputs: {diff:.2e}")
    if not np.array_equal(before.argmax(1), after.argmax(1)) or diff > 1e-6:
        args.target.unlink()
        sys.exit("Outputs differ; the slim file was removed.")


if __name__ == "__main__":
    main()
