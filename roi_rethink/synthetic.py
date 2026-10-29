"""Synthetic anomaly dataset.

BrReMark uses SynthSeg-generated synthetic brain MRIs in Phase 2 of GRPO
to scale training without clinical data, and masks the LLM-judge reward
on synthetic data to avoid hallucinating clinical correctness.

Here we play the same trick at toy scale: generate synthetic "scans"
(noisy textured canvases) with a planted anomalous blob of known shape,
color, and location, together with a ground-truth bounding box. This
gives us:
  * unlimited labeled data for the localization (IoU) reward,
  * a known binary label (anomaly present / absent),
  * a controlled test of whether the two-turn mark->rethink loop improves
    grounding over a single-turn answer.

Everything is PIL-only, runs in milliseconds, and is fully deterministic
given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter


@dataclass
class AnomalySample:
    image: Image.Image
    has_anomaly: bool
    bbox: Optional[Tuple[int, int, int, int]]  # ground truth, None if no anomaly
    anomaly_kind: Optional[str]  # e.g. "lesion", "hemorrhage", "calcification"
    prompt: str  # the clinician-style query to ask the model


_ANOMALY_KINDS = [
    ("lesion", (220, 80, 80)),
    ("hemorrhage", (180, 30, 30)),
    ("calcification", (230, 220, 120)),
    ("edema", (120, 200, 160)),
]


def _noisy_background(width: int, height: int, rng: random.Random) -> Image.Image:
    """A soft, textured grayscale 'scan' background."""
    img = Image.new("L", (width, height), 140)
    px = img.load()
    for y in range(height):
        for x in range(width):
            v = 140 + int(30 * rng.random()) - 15
            # radial dimming toward edges for a scan-like vignette
            cx, cy = width / 2, height / 2
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / (max(width, height) / 2)
            v -= int(40 * max(0, d - 0.5))
            px[x, y] = max(20, min(220, v))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return img.convert("RGB")


def _plant_anomaly(
    img: Image.Image, rng: random.Random, min_size: int, max_size: int
) -> Tuple[Tuple[int, int, int, int], str]:
    """Plant an anomalous blob; return its ground-truth bbox and kind."""
    kind, color = rng.choice(_ANOMALY_KINDS)
    w, h = img.size
    bw = rng.randint(min_size, max_size)
    bh = rng.randint(min_size, max_size)
    x1 = rng.randint(20, w - bw - 20)
    y1 = rng.randint(20, h - bh - 20)
    x2, y2 = x1 + bw, y1 + bh

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shape = rng.choice(["ellipse", "rect", "blob"])
    if shape == "ellipse":
        draw.ellipse([x1, y1, x2, y2], fill=(*color, 200))
    elif shape == "rect":
        draw.rectangle([x1, y1, x2, y2], fill=(*color, 200))
    else:
        # irregular blob: several overlapping circles
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for _ in range(6):
            r = rng.randint(bw // 4, bw // 2)
            ox = rng.randint(-bw // 3, bw // 3)
            oy = rng.randint(-bh // 3, bh // 3)
            draw.ellipse([cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r], fill=(*color, 190))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    img.paste(overlay, mask=overlay.split()[3])
    # Tighten GT bbox to the planted region with a small margin.
    pad = 4
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)), kind


class SyntheticAnomalyDataset:
    """Generates synthetic scan images with planted anomalies + GT bboxes."""

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        anomaly_prob: float = 0.7,
        min_size: int = 24,
        max_size: int = 64,
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.anomaly_prob = anomaly_prob
        self.min_size = min_size
        self.max_size = max_size
        self.seed = seed

    def generate(self, n: int, start_seed: Optional[int] = None) -> List[AnomalySample]:
        return list(self.iter(n, start_seed))

    def iter(self, n: int, start_seed: Optional[int] = None) -> Iterator[AnomalySample]:
        base = start_seed if start_seed is not None else self.seed
        for i in range(n):
            rng = random.Random(base + i)
            img = _noisy_background(self.width, self.height, rng)
            has = rng.random() < self.anomaly_prob
            bbox, kind = (None, None)
            if has:
                bbox, kind = _plant_anomaly(img, rng, self.min_size, self.max_size)
            prompt = (
                "Assess this scan for any suspected anomaly. "
                "If present, mark its location with mark_bbox and describe it."
            )
            yield AnomalySample(
                image=img, has_anomaly=has, bbox=bbox, anomaly_kind=kind, prompt=prompt
            )
