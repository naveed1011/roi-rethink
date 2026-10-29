"""Utility helpers: bounding-box IoU, tool-call parsing, and ROI marking.

These implement the same primitives BrReMark uses for its localization
reward and its `mark_bbox` environment feedback, but on plain PIL images
so the whole loop runs on a laptop.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageDraw

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2), absolute pixels


@dataclass
class ParsedMarkCall:
    """Result of parsing a model's <tool_call> ... </tool_call> block."""

    bbox: Optional[BBox]
    label: Optional[str]
    raw: str
    error: Optional[str] = None


def normalize_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Clamp a bbox into [0, width] x [0, height] and enforce x1<x2, y1<y2."""
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    # Degenerate boxes (zero area) are allowed but IoU will be 0.
    return (int(x1), int(y1), int(x2), int(y2))


def bbox_area(bbox: BBox) -> int:
    """Area of an axis-aligned bounding box in pixels."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


# A box smaller than 4x4 px cannot plausibly localize a lesion. The system
# prompt asks the model to emit [0, 0, 1, 1] as a placeholder when it sees no
# anomaly, so we treat such degenerate boxes as "no ROI marked".
MIN_MEANINGFUL_BBOX_AREA = 16


def is_meaningful_bbox(bbox: Optional[BBox]) -> bool:
    """True if `bbox` is a real ROI (not None and not a tiny sentinel box)."""
    return bbox is not None and bbox_area(bbox) >= MIN_MEANINGFUL_BBOX_AREA


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_mark_call(text: str) -> ParsedMarkCall:
    """Extract a mark_bbox tool call from model output.

    BrReMark's models emit:
        <tool_call> {"name": "mark_bbox", "arguments": {"bbox_2d": [...], "label": "..."}} </tool_call>
    We are lenient: we accept any JSON object with a bbox-like list of 4
    numbers under keys `bbox_2d`, `bbox`, or `coordinates`, plus an
    optional `label`.
    """
    m = _TOOL_CALL_RE.search(text)
    raw = m.group(1).strip() if m else text.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return ParsedMarkCall(bbox=None, label=None, raw=raw, error=f"JSON parse error: {e}")

    # The payload may be nested under "arguments" (OpenAI tool style) or flat.
    args = obj.get("arguments", obj)
    bbox = None
    for key in ("bbox_2d", "bbox", "coordinates", "box"):
        if key in args:
            vals = args[key]
            if isinstance(vals, (list, tuple)) and len(vals) == 4:
                try:
                    bbox = tuple(int(round(float(v))) for v in vals)  # type: ignore[assignment]
                except (TypeError, ValueError):
                    return ParsedMarkCall(
                        bbox=None, label=None, raw=raw, error="bbox values not numeric"
                    )
            break
    if bbox is None:
        return ParsedMarkCall(bbox=None, label=None, raw=raw, error="no 4-number bbox found")

    label = args.get("label") or obj.get("label")
    if label is not None:
        label = str(label)
    return ParsedMarkCall(bbox=bbox, label=label, raw=raw)  # type: ignore[arg-type]


def draw_marked_image(
    image: Image.Image,
    bbox: BBox,
    label: Optional[str] = None,
    color: str = "red",
    width: int = 4,
) -> Image.Image:
    """Return a copy of `image` with the ROI box drawn on it.

    This is exactly the feedback the BrReMark environment returns to the
    model on turn 2: the same image, with the hypothesized ROI highlighted.
    """
    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        # Simple label pill above the box.
        try:
            text_bbox = draw.textbbox((0, 0), label)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except Exception:
            tw, th = len(label) * 7, 14
        pad = 4
        ly = max(0, y1 - th - 2 * pad)
        draw.rectangle([x1, ly, x1 + tw + 2 * pad, ly + th + 2 * pad], fill=color)
        draw.text((x1 + pad, ly + pad), label, fill="white")
    return marked
