"""Composite reward, mirroring BrReMark's GRPO reward design.

BrReMark's Phase-1 reward is a weighted sum of:
  1. format compliance  (did the model emit the right <think>/<tool_call>/
     <rethink>/<answer> structure?),
  2. IoU localization   (does the marked box overlap the GT lesion?),
  3. LLM-as-judge       (clinical quality of the final answer, 0-10).

In Phase-2 (synthetic data) the LLM-judge term is *masked* because
synthetic scans have no clinical truth to judge against.

We implement 1 and 2 exactly, and expose 3 as an optional pluggable
`judge_fn` so you can wire in any scorer (an LLM, a keyword heuristic,
or a human). On synthetic data, leave judge_fn=None and it is masked,
exactly like BrReMark Phase 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from .utils import BBox, bbox_iou, is_meaningful_bbox, parse_mark_call

JudgeFn = Callable[[str, Optional[str]], float]  # (final_answer, reference) -> score in [0,1]


@dataclass
class RewardBreakdown:
    format_score: float       # [0,1]
    iou_score: float          # [0,1]
    judge_score: Optional[float]  # [0,1] or None if masked
    total: float              # weighted sum
    format_details: dict


def _format_score(turn1: str, turn2: str, expect_anomaly: bool) -> tuple[float, dict]:
    """Check the two-turn trajectory structure.

    BrReMark expects:
      turn1: <think> ... </think> then <tool_call> ... </tool_call>
      turn2: <rethink> ... </rethink> then <answer> ... </answer>
    Each present tag contributes. A valid, parseable mark_bbox call is the
    single most important term and gets the largest weight.
    """
    details = {}
    details["has_think"] = bool(re.search(r"<think>.*?</think>", turn1, re.DOTALL))
    has_tool = bool(re.search(r"<tool_call>.*?</tool_call>", turn1, re.DOTALL))
    details["has_tool_call"] = has_tool
    parsed = parse_mark_call(turn1)
    details["tool_call_valid"] = parsed.error is None and parsed.bbox is not None
    details["has_rethink"] = bool(re.search(r"<rethink>.*?</rethink>", turn2, re.DOTALL))
    has_answer = bool(re.search(r"<answer>.*?</answer>", turn2, re.DOTALL))
    details["has_answer"] = has_answer

    # If no anomaly expected, a model that correctly refuses to mark can
    # still get partial format credit; but we keep the same schema.
    weights = {"has_think": 0.10, "has_tool_call": 0.15, "tool_call_valid": 0.35,
               "has_rethink": 0.15, "has_answer": 0.25}
    score = sum(weights[k] for k, v in details.items() if v)
    return score, details


def composite_reward(
    turn1: str,
    turn2: str,
    gt_bbox: Optional[BBox],
    has_anomaly: bool,
    judge_fn: Optional[JudgeFn] = None,
    judge_reference: Optional[str] = None,
    w_format: float = 0.3,
    w_iou: float = 0.5,
    w_judge: float = 0.2,
    iou_threshold: float = 0.0,
) -> RewardBreakdown:
    """Compute the composite reward for one two-turn trajectory.

    gt_bbox:        ground-truth anomaly box (None if no anomaly).
    has_anomaly:    whether an anomaly is present.
    judge_fn:       optional LLM/human judge; if None, the judge term is
                    *masked* (BrReMark Phase-2 / synthetic-data behavior)
                    and weights are renormalized over format + IoU.
    iou_threshold:  if >0, IoU below threshold is floored to 0 (strict
                    localization; set 0.5 for "correct detection" style).
    """
    fmt, details = _format_score(turn1, turn2, has_anomaly)

    parsed = parse_mark_call(turn1)
    iou = 0.0
    if has_anomaly and gt_bbox is not None and parsed.bbox is not None:
        iou = bbox_iou(parsed.bbox, gt_bbox)
        if iou < iou_threshold:
            iou = 0.0
    elif not has_anomaly:
        # No anomaly: reward *not* marking a box (correct rejection).
        # A tiny placeholder box ([0,0,1,1], as requested by the system
        # prompt) counts as "no ROI marked", not as a false positive.
        iou = 0.0 if is_meaningful_bbox(parsed.bbox) else 1.0

    judge = None
    if judge_fn is not None:
        m = re.search(r"<answer>(.*?)</answer>", turn2, re.DOTALL)
        final_answer = m.group(1).strip() if m else turn2.strip()
        judge = max(0.0, min(1.0, judge_fn(final_answer, judge_reference)))

    if judge is None:
        # Mask judge term; renormalize format + IoU weights.
        denom = w_format + w_iou
        total = (w_format * fmt + w_iou * iou) / denom if denom > 0 else 0.0
    else:
        total = w_format * fmt + w_iou * iou + w_judge * judge

    return RewardBreakdown(
        format_score=fmt,
        iou_score=iou,
        judge_score=judge,
        total=total,
        format_details=details,
    )
