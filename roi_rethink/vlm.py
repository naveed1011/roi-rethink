"""VLM client abstraction.

BrReMark uses Lingshu-7B served via vLLM. For an educational project we
abstract the model behind a tiny interface so you can plug in *any*
OpenAI-compatible vision endpoint (OpenAI, Groq, Together, a local
vLLM/Ollama server, ...). A deterministic MockVLM is also provided so the
whole pipeline can be exercised and tested offline without an API key.
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image


class VLMClient(ABC):
    @abstractmethod
    def complete(self, messages: List[dict], **kwargs) -> str:
        """Send a chat message list (which may include image content) and
        return the assistant's text reply."""
        raise NotImplementedError


def _image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


class OpenAICompatibleVLM(VLMClient):
    """Thin client for any OpenAI-compatible /v1/chat/completions endpoint
    that supports image_url content parts.

    Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment,
    or pass them in.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai package is required for OpenAICompatibleVLM. "
                "Install with `pip install openai`."
            ) from e
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    def complete(self, messages: List[dict], **kwargs) -> str:
        # Normalize: if a message has an "image" key (PIL Image), convert to
        # an image_url content part.
        normalized = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content")
            image = msg.get("image")
            if image is not None:
                parts: List[dict] = []
                if content:
                    parts.append({"type": "text", "text": str(content)})
                parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": content})
        resp = self.client.chat.completions.create(
            model=self.model, messages=normalized, **kwargs
        )
        return resp.choices[0].message.content or ""


class MockVLM(VLMClient):
    """Deterministic offline stand-in.

    It does NOT actually see the image. It returns a scripted two-turn
    trajectory: on turn 1 it emits a mark_bbox call (using a bbox hint
    stored in the prompt, or a default), and on turn 2 it emits a final
    <answer>. Used for tests, demos, and to show the *loop* without a
    model. Real usage swaps in OpenAICompatibleVLM.

    The mock is *stateless*: it decides which turn it is on by inspecting
    the conversation (has an assistant already emitted a tool call?). This
    matters because `evaluate()` reuses one client across many samples — an
    internal turn counter would desynchronise after the first sample and
    silently skip every later mark.
    """

    def __init__(self, hint_bbox=(120, 120, 200, 200), final_answer="Anomaly detected."):
        self.hint_bbox = hint_bbox
        self.final_answer = final_answer

    def complete(self, messages: List[dict], **kwargs) -> str:
        if _is_final_answer_prompt(messages):
            return f"<answer>{self.final_answer}</answer>"
        if _already_marked(messages):
            return (
                "<rethink>With the ROI marked, the region is consistent with an anomaly.</rethink>\n"
                f"<answer>{self.final_answer}</answer>"
            )
        x1, y1, x2, y2 = self.hint_bbox
        return (
            "<think>Examining the image for anomalous regions.</think>\n"
            f'<tool_call>{{"name": "mark_bbox", "arguments": {{"bbox_2d": [{x1}, {y1}, {x2}, {y2}], "label": "suspected anomaly"}}}}</tool_call>'
        )


def _already_marked(messages: List[dict]) -> bool:
    """True once the assistant has emitted a tool call in this conversation."""
    return any(
        m.get("role") == "assistant" and "<tool_call>" in str(m.get("content") or "")
        for m in messages
    )


def _is_final_answer_prompt(messages: List[dict]) -> bool:
    """True on the agent's fallback path, which asks for a bare <answer>."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "").strip().startswith("Give your final")
    return False
