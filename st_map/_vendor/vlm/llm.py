import json
import re

from st_map._vendor.vlm.client import ChatClient

# object types of the UniNavBench houses: demand-driven targets are chosen among them
OBJECT_TYPES = ['alarm clock', 'apple', 'arm chair', 'baseball bat', 'basket ball', 'bed', 'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'butter knife', 'candle', 'cart', 'cd', 'cell phone', 'chair', 'cloth', 'clothes dryer', 'coffee machine', 'coffee table', 'counter top', 'credit card', 'cup', 'desk', 'desk lamp', 'desktop', 'dining table', 'dish sponge', 'dog bed', 'dresser', 'dumbbell', 'egg', 'faucet', 'floor lamp', 'fork', 'fridge', 'garbage bag', 'garbage can', 'house plant', 'kettle', 'key chain', 'knife', 'ladle', 'laptop', 'laundry hamper', 'lettuce', 'microwave', 'mug', 'newspaper', 'ottoman', 'painting', 'pan', 'paper towel roll', 'pen', 'pencil', 'pepper shaker', 'pillow', 'plate', 'plunger', 'pot', 'potato', 'remote control', 'room decor', 'safe', 'salt shaker', 'shelving unit', 'side table', 'sink', 'soap bar', 'soap bottle', 'sofa', 'spatula', 'spoon', 'spray bottle', 'statue', 'stool', 'table top decor', 'teddy bear', 'television', 'tennis racket', 'tissue box', 'toaster', 'toilet', 'toilet paper', 'tomato', 'tv stand', 'vacuum cleaner', 'vase', 'washing machine', 'watch', 'wine bottle']


class LLMQwQ():
    """Planner LLM: picks the next graph node from a compact text description of the subgraph."""

    def __init__(self, base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", model="default", **options):
        self.history = []
        self.model = model
        self.chat = ChatClient(base_url, api_key, model, **{"max_tokens": 64, **options})
        self._targets = {}  # instruction -> inferred target object

    def query_node_txt(self, user, note=None):
        from st_map._vendor.vl_prompt.prompt_gpt.queryNodetxt import SYSTEM_PROMPT
        if note:
            user = user + "\n" + note
        reply = self.chat.chat([{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}], "llm")
        reply = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        result = reply['result'] if isinstance(reply['result'], int) else int(re.findall(r"\d+", str(reply['result']))[0])
        reply.setdefault('reason', '')
        return reply, result

    def target_object(self, instruction):
        """The household object that would satisfy an instruction naming none ("I need a method of payment")."""
        if instruction not in self._targets:
            try:
                reply = self.chat.chat([{'role': 'user', 'content': "Which household object would satisfy this request? "
                                         f"Choose one from: {', '.join(OBJECT_TYPES)}.\nReply with the object name only.\n"
                                         f"Request: {instruction}"}], "llm", 8)
                self._targets[instruction] = reply.strip().strip(".").lower() or None
            except Exception as ex:
                print("target object query failed:", ex)
                self._targets[instruction] = None
        return self._targets[instruction]
