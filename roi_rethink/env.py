"""The two-turn mark_bbox environment.

This is the heart of BrReMark's "ROI rethink" mechanism, isolated:

  Turn 1: the model sees the raw image and emits a hypothesis in the form
          of a <tool_call>mark_bbox</tool_call> with a bounding box.
  Env:    validates the call, draws the box on the image, and returns the
          *marked* image as observation.
  Turn 2: the model re-examines the marked ROI ("rethink") and produces a
          final grounded <answer>.

The environment is intentionally model-agnostic; it only knows about the
mark_bbox tool contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from .utils import (
    BBox,
    ParsedMarkCall,
    draw_marked_image,
    is_meaningful_bbox,
    normalize_bbox,
    parse_mark_call,
)


@dataclass
class MarkResult:
    """Outcome of processing the model's turn-1 output."""

    parsed: ParsedMarkCall
    marked_image: Optional[Image.Image] = None
    valid: bool = False
    feedback: str = ""


@dataclass
class TwoTurnMarkEnv:
    """Stateful two-turn environment for a single image."""

    image: Image.Image
    mark_color: str = "red"
    mark_width: int = 4
    _marked: Optional[Image.Image] = field(default=None, init=False)
    _last_bbox: Optional[BBox] = field(default=None, init=False)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def tool_spec(self) -> dict:
        """A textual tool spec to inject into the system prompt."""
        return {
            "name": "mark_bbox",
            "description": (
                "Mark a region of interest (ROI) on the image for re-examination. "
                "Returns the same image with the box drawn on it."
            ),
            "parameters": {
                "bbox_2d": f"[x1, y1, x2, y2] in absolute pixels, image is {self.width}x{self.height}",
                "label": "short label for the marked region",
            },
        }

    def step(self, model_output: str) -> MarkResult:
        """Process turn-1 model output. Returns a MarkResult; if valid,
        `marked_image` is the new observation for turn 2."""
        parsed = parse_mark_call(model_output)
        if parsed.error or parsed.bbox is None:
            return MarkResult(
                parsed=parsed,
                valid=False,
                feedback=(
                    "Invalid mark_bbox call. "
                    f"Reason: {parsed.error or 'missing bbox'}. "
                    "Emit exactly one <tool_call>...</tool_call> block with "
                    '{"name": "mark_bbox", "arguments": {"bbox_2d": [x1,y1,x2,y2], "label": "..."}}.'
                ),
            )
        bbox = normalize_bbox(parsed.bbox, self.width, self.height)
        self._last_bbox = bbox
        if not is_meaningful_bbox(bbox):
            # Placeholder box (typically [0,0,1,1] + label "no anomaly").
            # Keep the image unchanged so turn 2 still has an observation,
            # but tell the model explicitly that no ROI was marked.
            self._marked = self.image.convert("RGB").copy()
            return MarkResult(
                parsed=parsed,
                marked_image=self._marked,
                valid=True,
                feedback=(
                    "No ROI marked (placeholder box received). "
                    "If the scan is normal, state that in your final <answer>."
                ),
            )
        self._marked = draw_marked_image(
            self.image, bbox, label=parsed.label, color=self.mark_color, width=self.mark_width
        )
        return MarkResult(
            parsed=parsed,
            marked_image=self._marked,
            valid=True,
            feedback=f"ROI marked at {bbox}. Re-examine the highlighted region and give your final <answer>.",
        )

    @property
    def last_bbox(self) -> Optional[BBox]:
        return self._last_bbox
