import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402

from roi_rethink.env import TwoTurnMarkEnv  # noqa: E402


def test_env_valid_mark():
    img = Image.new("RGB", (100, 100), "gray")
    env = TwoTurnMarkEnv(img)
    out = '<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [10,10,50,50], "label": "x"}}</tool_call>'
    res = env.step(out)
    assert res.valid
    assert res.marked_image is not None
    assert res.marked_image.size == (100, 100)
    assert env.last_bbox == (10, 10, 50, 50)


def test_env_invalid_mark_feedback():
    img = Image.new("RGB", (100, 100), "gray")
    env = TwoTurnMarkEnv(img)
    res = env.step("no tool call at all")
    assert not res.valid
    assert "Invalid" in res.feedback
    assert env.last_bbox is None


def test_env_bbox_clamped():
    img = Image.new("RGB", (50, 50), "gray")
    env = TwoTurnMarkEnv(img)
    out = '<tool_call>{"arguments": {"bbox_2d": [-10, -10, 200, 200]}}</tool_call>'
    res = env.step(out)
    assert res.valid
    assert env.last_bbox == (0, 0, 50, 50)


def test_env_placeholder_box_leaves_image_unchanged():
    # The system prompt tells the model to emit [0,0,1,1] when it sees no
    # anomaly; that must not draw a visible ROI.
    img = Image.new("RGB", (20, 20), "gray")
    env = TwoTurnMarkEnv(img)
    out = (
        '<tool_call>{"name": "mark_bbox", "arguments": '
        '{"bbox_2d": [0, 0, 1, 1], "label": "no anomaly"}}</tool_call>'
    )
    res = env.step(out)
    assert res.valid
    assert res.marked_image is not None
    assert list(res.marked_image.getdata()) == list(img.convert("RGB").getdata())
    assert "No ROI marked" in res.feedback
