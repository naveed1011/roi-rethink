import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.reward import composite_reward  # noqa: E402
from roi_rethink.utils import bbox_iou, parse_mark_call  # noqa: E402


def test_iou_identical():
    assert abs(bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9


def test_iou_disjoint():
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # [0,20]x[0,10] vs [10,30]x[0,10]: inter=100, union=300
    iou = bbox_iou((0, 0, 20, 10), (10, 0, 30, 10))
    assert abs(iou - 1 / 3) < 1e-9


def test_parse_mark_call_valid():
    text = '<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [10, 20, 30, 40], "label": "lesion"}}</tool_call>'
    p = parse_mark_call(text)
    assert p.error is None
    assert p.bbox == (10, 20, 30, 40)
    assert p.label == "lesion"


def test_parse_mark_call_bad_json():
    p = parse_mark_call("<tool_call>not json</tool_call>")
    assert p.error is not None
    assert p.bbox is None


def test_reward_perfect_trajectory():
    turn1 = (
        "<think>blob in upper right</think>\n"
        '<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [10,10,20,20], "label": "lesion"}}</tool_call>'
    )
    turn2 = "<rethink>confirmed</rethink>\n<answer>anomaly present</answer>"
    r = composite_reward(turn1, turn2, gt_bbox=(10, 10, 20, 20), has_anomaly=True)
    assert r.format_score == 1.0
    assert r.iou_score == 1.0
    assert r.judge_score is None  # masked by default
    assert abs(r.total - 1.0) < 1e-9


def test_reward_judge_masked_on_synthetic():
    # With no judge_fn, judge term is masked and weights renormalize.
    turn1 = '<tool_call>{"arguments": {"bbox_2d": [0,0,1,1]}}</tool_call>'
    turn2 = "<answer>ok</answer>"
    r = composite_reward(turn1, turn2, gt_bbox=None, has_anomaly=False)
    assert r.judge_score is None
    assert 0.0 <= r.total <= 1.0


def test_reward_correct_rejection():
    # No anomaly, model marks nothing -> IoU term = 1.0
    turn1 = "<think>clear</think>\n<tool_call>{}</tool_call>"
    turn2 = "<rethink>clear</rethink>\n<answer>no anomaly</answer>"
    r = composite_reward(turn1, turn2, gt_bbox=None, has_anomaly=False)
    # malformed tool call lowers format, but correct-rejection IoU is 1
    assert r.iou_score == 1.0


def test_reward_placeholder_box_counts_as_correct_rejection():
    # The system prompt asks for [0,0,1,1] + "no anomaly" on normal scans.
    # That must score as a correct rejection, not as a false positive.
    turn1 = (
        "<think>no lesion visible</think>\n"
        '<tool_call>{"name": "mark_bbox", "arguments": '
        '{"bbox_2d": [0, 0, 1, 1], "label": "no anomaly"}}</tool_call>'
    )
    turn2 = "<rethink>still clear</rethink>\n<answer>No anomaly.</answer>"
    r = composite_reward(turn1, turn2, gt_bbox=None, has_anomaly=False)
    assert r.format_score == 1.0
    assert r.iou_score == 1.0
    assert r.total == 1.0


def test_reward_real_box_on_normal_scan_is_false_positive():
    turn1 = (
        "<think>maybe</think>\n"
        '<tool_call>{"arguments": {"bbox_2d": [10, 10, 40, 40]}}</tool_call>'
    )
    turn2 = "<rethink>hmm</rethink>\n<answer>Anomaly.</answer>"
    r = composite_reward(turn1, turn2, gt_bbox=None, has_anomaly=False)
    assert r.iou_score == 0.0
