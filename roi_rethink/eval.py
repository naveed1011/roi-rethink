"""Evaluation harness.

Runs the two-turn agent over a dataset, computes the composite reward per
sample (format + IoU + optional judge), and aggregates:
  * mean total reward, format, IoU,
  * detection accuracy (anomaly present/absent),
  * IoU@0.5 (standard detection success rate),
  * per-sample JSON records for analysis.

This mirrors BrReMark's eval/README "LLM-as-judge evaluation" but keeps
the deterministic metrics (format, IoU) first-class, with the judge as an
optional plug-in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from tqdm import tqdm

from .agent import RethinkAgent
from .reward import JudgeFn, RewardBreakdown, composite_reward
from .synthetic import AnomalySample
from .utils import is_meaningful_bbox


@dataclass
class EvalRecord:
    sample_index: int
    has_anomaly: bool
    anomaly_kind: Optional[str]
    gt_bbox: Optional[tuple]
    marked_bbox: Optional[tuple]
    valid_mark: bool
    predicted_positive: bool
    final_answer: str
    turn1: str
    turn2: str
    reward: RewardBreakdown


def evaluate(
    agent: RethinkAgent,
    samples: List[AnomalySample],
    judge_fn: Optional[JudgeFn] = None,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    records: List[EvalRecord] = []
    iterator = enumerate(samples)
    if verbose:
        iterator = enumerate(tqdm(samples, desc="evaluating"))
    for i, sample in iterator:
        traj = agent.run(sample.image, sample.prompt)
        reward = composite_reward(
            traj.turn1, traj.turn2,
            gt_bbox=sample.bbox,
            has_anomaly=sample.has_anomaly,
            judge_fn=judge_fn,
            judge_reference=sample.anomaly_kind,
            iou_threshold=0.0,  # raw IoU in reward; threshold applied for metrics
        )
        records.append(EvalRecord(
            sample_index=i,
            has_anomaly=sample.has_anomaly,
            anomaly_kind=sample.anomaly_kind,
            gt_bbox=sample.bbox,
            marked_bbox=traj.marked_bbox,
            valid_mark=traj.valid_mark,
            predicted_positive=is_meaningful_bbox(traj.marked_bbox),
            final_answer=traj.final_answer,
            turn1=traj.turn1,
            turn2=traj.turn2,
            reward=reward,
        ))

    n = len(records)
    positives = [r for r in records if r.has_anomaly]
    metrics = {
        "n": n,
        "mean_reward": sum(r.reward.total for r in records) / n if n else 0.0,
        "mean_format": sum(r.reward.format_score for r in records) / n if n else 0.0,
        "mean_iou": sum(r.reward.iou_score for r in records) / n if n else 0.0,
        "mean_iou_positive": (
            sum(r.reward.iou_score for r in positives) / len(positives) if positives else 0.0
        ),
        "valid_mark_rate": sum(r.valid_mark for r in records) / n if n else 0.0,
        "detection_accuracy": (
            sum(r.predicted_positive == r.has_anomaly for r in records) / n if n else 0.0
        ),
        "iou_at_0_5": (
            sum(1 for r in positives if r.reward.iou_score >= iou_threshold)
            / max(1, len(positives))
        ),
        "anomaly_prevalence": sum(r.has_anomaly for r in records) / n if n else 0.0,
    }
    judge_scores = [r.reward.judge_score for r in records if r.reward.judge_score is not None]
    if judge_scores:
        metrics["mean_judge"] = sum(judge_scores) / len(judge_scores)

    return {
        "metrics": metrics,
        "records": [_record_to_dict(r) for r in records],
    }


def _record_to_dict(r: EvalRecord) -> dict:
    d = asdict(r)
    d["reward"] = {
        "format_score": r.reward.format_score,
        "iou_score": r.reward.iou_score,
        "judge_score": r.reward.judge_score,
        "total": r.reward.total,
        "format_details": r.reward.format_details,
    }
    return d


def save_eval(result: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
