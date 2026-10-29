import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402


def test_generate_deterministic():
    ds = SyntheticAnomalyDataset(seed=42)
    a = ds.generate(5)
    b = ds.generate(5)
    assert len(a) == len(b) == 5
    for sa, sb in zip(a, b):
        assert sa.has_anomaly == sb.has_anomaly
        assert sa.bbox == sb.bbox
        assert list(sa.image.getdata()) == list(sb.image.getdata())


def test_gt_bbox_in_bounds():
    ds = SyntheticAnomalyDataset(width=200, height=180, seed=1)
    for s in ds.generate(20):
        assert s.image.size == (200, 180)
        if s.bbox is not None:
            x1, y1, x2, y2 = s.bbox
            assert 0 <= x1 < x2 <= 200
            assert 0 <= y1 < y2 <= 180


def test_anomaly_mix():
    ds = SyntheticAnomalyDataset(anomaly_prob=0.5, seed=0)
    samples = ds.generate(40)
    pos = sum(s.has_anomaly for s in samples)
    # With p=0.5 and n=40, expect roughly half; allow slack.
    assert 10 <= pos <= 30
