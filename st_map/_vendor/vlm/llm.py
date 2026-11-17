import json
import re

from st_map._vendor.vlm.client import ChatClient


class LLMQwQ():
    """Planner LLM: picks the next graph node from a compact text description of the subgraph."""

    def __init__(self, base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model="default", **options):
        self.history = []
        self.model = model
        self.chat = ChatClient(base_url, api_key, model, **{"max_tokens": 64, **options})

    def query_node_txt(self, user):
        from st_map._vendor.vl_prompt.prompt_gpt.queryNodetxt import SYSTEM_PROMPT
        reply = self.chat.chat([{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}], "llm")
        reply = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        result = reply['result'] if isinstance(reply['result'], int) else int(re.findall(r"\d+", str(reply['result']))[0])
        reply.setdefault('reason', '')
        return reply, result
