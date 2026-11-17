"""OpenAI-compatible chat client shared by the planner LLM and the verifier VLM.

Thinking is off by default (it multiplies output tokens); DashScope only streams thinking replies, so a
thinking request is streamed and collected. Token usage is summed per call kind in USAGE.
"""
import base64
import io
from collections import Counter

from openai import OpenAI
from PIL import Image

USAGE = Counter()  # "<kind>_calls", "<kind>_prompt_tokens", "<kind>_completion_tokens"


class ChatClient:
    def __init__(self, base_url=None, api_key=None, model="default", enable_thinking=False, max_tokens=None,
                 timeout=120, **_):
        self.model = model
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=timeout, max_retries=2)

    def chat(self, messages, kind, max_tokens=None):
        kwargs = dict(model=self.model, messages=messages, extra_body={"enable_thinking": bool(self.enable_thinking)})
        if (max_tokens or self.max_tokens) and not self.enable_thinking:
            kwargs["max_tokens"] = max_tokens or self.max_tokens
        USAGE[kind + "_calls"] += 1
        if not self.enable_thinking:
            reply = self.client.chat.completions.create(**kwargs)
            self._count(kind, reply.usage)
            return reply.choices[0].message.content or ""
        parts = []
        for chunk in self.client.chat.completions.create(stream=True, stream_options={"include_usage": True}, **kwargs):
            if chunk.usage is not None:
                self._count(kind, chunk.usage)
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts)

    @staticmethod
    def _count(kind, usage):
        if usage is not None:
            USAGE[kind + "_prompt_tokens"] += usage.prompt_tokens or 0
            USAGE[kind + "_completion_tokens"] += usage.completion_tokens or 0


def image_url(path, box=None, max_side=320, margin=0.3):
    """JPEG data URL of an image, cropped to box (x1, y1, x2, y2) plus a margin and scaled to max_side.

    A crop around the object is both clearer for the verifier and cheaper: Qwen-VL bills about 100 tokens
    for an image of at most 320 px versus about 335 for a full 640x480 frame.
    """
    img = Image.open(path).convert("RGB")
    if box is not None:
        x1, y1, x2, y2 = [float(v) for v in box]
        mx, my = (x2 - x1) * margin + 16, (y2 - y1) * margin + 16
        img = img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)), min(img.width, int(x2 + mx)), min(img.height, int(y2 + my))))
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}
