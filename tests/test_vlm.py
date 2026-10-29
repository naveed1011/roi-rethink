import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM  # noqa: E402


def test_mock_is_stateless_across_eval_samples():
    """Regression test: the mock used to keep a global turn counter, so only
    the first evaluated sample produced a valid mark."""
    samples = SyntheticAnomalyDataset(seed=1).generate(3)
    agent = RethinkAgent(MockVLM(hint_bbox=(10, 10, 40, 40)))
    for sample in samples:
        traj = agent.run(sample.image, sample.prompt)
        assert traj.valid_mark
        assert traj.marked_bbox == (10, 10, 40, 40)
        assert "<answer>" in traj.turn2


def test_mock_answers_fallback_prompt():
    out = MockVLM().complete(
        [{"role": "user", "content": "Give your final <answer> now."}]
    )
    assert out == "<answer>Anomaly detected.</answer>"


def test_openai_client_normalizes_image_content():
    pytest.importorskip("openai")
    from roi_rethink import vlm as vlm_mod

    client = vlm_mod.OpenAICompatibleVLM(api_key="sk-test")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class Message:
                content = "ok"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    client.client = FakeClient()
    out = client.complete(
        [{"role": "user", "content": "hi", "image": Image.new("RGB", (8, 8), "red")}],
        max_tokens=16,
    )

    assert out == "ok"
    parts = captured["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "hi"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["max_tokens"] == 16
