import os

from st_map._vendor.vlm.client import ChatClient, image_url

FIND = "Consider the instruction: '{}'. What specific object is this instruction asking to find? Is that object visible in {}? Answer only True or False."
SAME = "Do {} show the same physical object referred to by the instruction '{}', in the same environment, as the last image? Answer only True or False."
ANSWER = ("You are a robot answering a question about the house you are in, from what you see now and from earlier "
          "close-up views. Question: {}\nAnswer in at most 10 words.")


class VLMQwQ():
    """Verifier VLM. Images are cropped to the candidate object and scaled down (see client.image_url)."""

    def __init__(self, base_url=None, api_key=None, model="default", image_max_side=320, crop_margin=0.3, **options):
        self.history = []
        self.model = model
        self.max_side, self.margin = image_max_side, crop_margin
        self.chat = ChatClient(base_url or os.environ.get("ST_MAP_VLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                               api_key or os.environ.get("ST_MAP_VLM_API_KEY", "EMPTY"), model, **{"max_tokens": 8, **options})

    def _img(self, path, box=None):
        return image_url(path, box, self.max_side, self.margin)

    def _ask(self, content, kind, max_tokens=None):
        try:
            return self.chat.chat([{'role': 'user', 'content': content}], kind, max_tokens)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def query_node_txt(self, img_name):
        return self._ask([{"type": "text", "text": "Describe the image in at most 40 words."}, self._img(img_name)], "caption", 80)

    def query_multi_images(self, img_list, obj_name):
        content = [{"type": "text", "text": f"Describe the {obj_name} in these views in at most 40 words: color, material, "
                                            "what it is on or next to, and any change between views."}]
        content += [self._img(img) for img in img_list]
        return self._ask(content, "caption", 80)

    def judge_obj_in_img(self, img, obj_name, img_task, box=None):
        return self.judge_obj_in_imglist([img], obj_name, img_task, [box])

    def judge_obj_in_imglist(self, img_list, obj_name, img_task, boxes=None):
        boxes = boxes or [None] * len(img_list)
        views = "this image" if len(img_list) == 1 else "these images"
        if img_task is None:
            content = [{"type": "text", "text": FIND.format(obj_name, views)}]
        else:
            content = [{"type": "text", "text": SAME.format(views if len(img_list) > 1 else "the first image", obj_name)}]
        content += [self._img(img, box) for img, box in zip(img_list, boxes)]
        if img_task is not None:
            content.append(self._img(img_task))
        return self._ask(content, "verify")

    def answer(self, question, current_frame, views=()):
        """Active-EQA answer from the current frame and close-ups [(image path, box)] of the most related object."""
        content = [{"type": "text", "text": ANSWER.format(question)}, self._img(current_frame)]
        content += [self._img(path, box) for path, box in views]
        return self._ask(content, "answer", 24)
