"""Minimal end-to-end example: generate one synthetic scan, run the
two-turn agent (offline MockVLM), and print the reward.

Run:  python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.reward import composite_reward  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM  # noqa: E402


def main():
    sample = next(SyntheticAnomalyDataset(seed=5).iter(1))
    print(f"GT: has_anomaly={sample.has_anomaly}, kind={sample.anomaly_kind}, bbox={sample.bbox}")

    agent = RethinkAgent(MockVLM(hint_bbox=sample.bbox or (10, 10, 20, 20)))
    traj = agent.run(sample.image, sample.prompt)

    print("\n--- Turn 1 ---")
    print(traj.turn1)
    print("\n--- Marked bbox:", traj.marked_bbox, "(valid:", traj.valid_mark, ")")
    print("\n--- Turn 2 ---")
    print(traj.turn2)

    reward = composite_reward(traj.turn1, traj.turn2, sample.bbox, sample.has_anomaly)
    print("\n--- Reward ---")
    print(f"format={reward.format_score:.2f}  iou={reward.iou_score:.2f}  "
          f"judge={reward.judge_score}  total={reward.total:.3f}")


if __name__ == "__main__":
    main()
