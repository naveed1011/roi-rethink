"""roi-rethink: a lightweight re-implementation of the two-turn
'hypothesis -> mark ROI -> rethink -> answer' visual reasoning loop
inspired by BrReMark (Li et al., 2026).

The goal of this package is educational: it isolates the *mechanism* of
grounded visual reasoning (tool-use marking + a second-turn re-examination
of the marked ROI) and the *composite reward* design (format compliance +
IoU localization + optional judge) so they can be studied, run, and
evaluated on commodity hardware with any OpenAI-compatible VLM endpoint.
"""

__version__ = "0.1.0"

from .env import TwoTurnMarkEnv, MarkResult
from .reward import composite_reward, RewardBreakdown
from .synthetic import SyntheticAnomalyDataset, AnomalySample
from .utils import (
    bbox_iou,
    parse_mark_call,
    draw_marked_image,
    normalize_bbox,
)

__all__ = [
    "__version__",
    "TwoTurnMarkEnv",
    "MarkResult",
    "composite_reward",
    "RewardBreakdown",
    "SyntheticAnomalyDataset",
    "AnomalySample",
    "bbox_iou",
    "parse_mark_call",
    "draw_marked_image",
    "normalize_bbox",
]
