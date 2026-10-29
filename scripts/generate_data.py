#!/usr/bin/env python3
"""Generate a synthetic anomaly dataset (images + metadata JSON).

Analogous to BrReMark's SynthSeg-based Phase-2 data, but toy-scale and
fully self-contained. Output:
  data/synthetic/images/sample_XXXX.png
  data/synthetic/metadata.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="number of samples")
    p.add_argument("--out", type=str, default="data/synthetic")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anomaly-prob", type=float, default=0.7)
    args = p.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    ds = SyntheticAnomalyDataset(
        width=args.size, height=args.size,
        anomaly_prob=args.anomaly_prob, seed=args.seed,
    )
    meta = []
    for i, sample in enumerate(ds.iter(args.n)):
        name = f"sample_{i:04d}.png"
        sample.image.save(os.path.join(img_dir, name))
        meta.append({
            "image": f"images/{name}",
            "has_anomaly": sample.has_anomaly,
            "anomaly_kind": sample.anomaly_kind,
            "bbox": sample.bbox,
            "prompt": sample.prompt,
        })

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Generated {args.n} samples -> {args.out}")


if __name__ == "__main__":
    main()
