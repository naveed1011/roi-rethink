"""The two-turn reasoning agent.

Orchestrates: query + image -> model turn1 (hypothesis + mark_bbox) ->
env step (draw ROI) -> model turn2 (rethink marked image + final answer).

This is the inference-side counterpart of BrReMark's two-turn trajectory.
BrReMark trains the model to emit this structure; here we *enforce* the
structure at inference time on any VLM via prompting, so you can study the
mechanism without training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .env import TwoTurnMarkEnv
from .utils import BBox
from .vlm import VLMClient

SYSTEM_PROMPT = """You are a medical-imaging assistant following a grounded two-turn reasoning protocol.

TURN 1 (hypothesis): Look at the image carefully. First write your observation in <think>...</think> tags. Then, if you suspect an anomaly, call exactly one tool to mark its region:
<tool_call>{"name": "mark_bbox", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "short label"}}</tool_call>
If you see NO anomaly, still emit a <tool_call> with bbox_2d set to [0,0,1,1] and label "no anomaly".

TURN 2 (rethink): You will receive the same image with your marked ROI highlighted. Re-examine it in <rethink>...</rethink> tags, then give your final conclusion in <answer>...</answer> tags. Be concise and grounded in the marked region.
"""


@dataclass
class AgentTrajectory:
    turn1: str
    turn2: str
    marked_bbox: Optional[BBox]
    marked_image: Optional[Image.Image]
    final_answer: str
    valid_mark: bool


class RethinkAgent:
    def __init__(self, vlm: VLMClient, system_prompt: str = SYSTEM_PROMPT, max_retries: int = 1):
        self.vlm = vlm
        self.system_prompt = system_prompt
        self.max_retries = max_retries

    def run(self, image: Image.Image, query: str) -> AgentTrajectory:
        env = TwoTurnMarkEnv(image)
        messages = [
            {"role": "system", "content": self.system_prompt + f"\n\nTool spec: {env.tool_spec}"},
            {"role": "user", "content": query, "image": image},
        ]

        # Turn 1: hypothesis + mark. Retry once if the call is malformed.
        turn1 = ""
        result = None
        for attempt in range(self.max_retries + 1):
            turn1 = self.vlm.complete(messages, max_tokens=512)
            result = env.step(turn1)
            if result.valid:
                break
            messages.append({"role": "assistant", "content": turn1})
            messages.append({"role": "user", "content": result.feedback})

        if not result.valid or result.marked_image is None:
            # Fall through: treat turn2 as a direct answer on the raw image.
            messages.append({"role": "user", "content": "Give your final <answer> now."})
            turn2 = self.vlm.complete(messages, max_tokens=512)
            return AgentTrajectory(
                turn1=turn1, turn2=turn2, marked_bbox=None,
                marked_image=None, final_answer=_extract_answer(turn2), valid_mark=False,
            )

        # Turn 2: rethink on the marked image.
        messages.append({"role": "assistant", "content": turn1})
        messages.append({
            "role": "user",
            "content": result.feedback,
            "image": result.marked_image,
        })
        turn2 = self.vlm.complete(messages, max_tokens=512)
        return AgentTrajectory(
            turn1=turn1, turn2=turn2, marked_bbox=env.last_bbox,
            marked_image=result.marked_image, final_answer=_extract_answer(turn2),
            valid_mark=True,
        )


def _extract_answer(turn2: str) -> str:
    import re
    m = re.search(r"<answer>(.*?)</answer>", turn2, re.DOTALL)
    return m.group(1).strip() if m else turn2.strip()
