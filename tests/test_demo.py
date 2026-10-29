import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.vlm import MockVLM  # noqa: E402

_DEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "demo_gradio.py")


def test_gradio_demo_builds():
    pytest.importorskip("gradio")
    spec = importlib.util.spec_from_file_location("demo_gradio", _DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    demo = module.build_demo(RethinkAgent(MockVLM()))
    assert demo is not None
