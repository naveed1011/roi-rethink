import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.eval import evaluate  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM  # noqa: E402


def test_eval_metrics_with_blind_mock_are_consistent():
    samples = SyntheticAnomalyDataset(seed=3).generate(6)
    result = evaluate(
        RethinkAgent(MockVLM(hint_bbox=(0, 0, 30, 30))),
        samples,
        verbose=False,
    )
    m = result["metrics"]

    assert m["n"] == 6
    assert m["valid_mark_rate"] == 1.0
    assert m["mean_format"] == 1.0
    assert 0.0 <= m["mean_reward"] <= 1.0
    assert 0.0 <= m["mean_iou_positive"] <= 1.0
    # The blind mock marks every image, so it calls every normal scan positive;
    # its binary accuracy is therefore just the anomaly prevalence.
    assert abs(m["detection_accuracy"] - m["anomaly_prevalence"]) < 1e-9

    assert len(result["records"]) == 6
    assert all(r["predicted_positive"] for r in result["records"])
    assert result["records"][0]["turn1"].startswith("<think>")
    assert "<rethink>" in result["records"][0]["turn2"]


def test_eval_handles_empty_dataset():
    result = evaluate(RethinkAgent(MockVLM()), [], verbose=False)
    assert result["metrics"]["n"] == 0
    assert result["metrics"]["mean_reward"] == 0.0
    assert result["records"] == []
