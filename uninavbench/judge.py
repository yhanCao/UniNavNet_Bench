"""Scoring of Active-EQA answers: an LLM judge (any OpenAI-compatible endpoint) or exact matching."""
import json
import os
import re

SYSTEM_PROMPT = '''You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
5 means that the response perfectly matches the answer.
1 means that the response is completely different from the answer.
Only output the response and nothing else.
Example 1:
Input:
    Question: Is it overcast?
    Answer: no
    Response: yes
Output: 1

Example 2:
Input:
    Question: Who is standing at the table?
    Answer: woman
    Response: Jessica
Output:3

Example 3:
Input:
    Question: Are there drapes to the right of the bed?
    Answer: yes
    Response: yes
Output:5

'''


class LLMJudge:
    """Scores a response from 1 to 5 with an LLM behind an OpenAI-compatible chat API.

    Settings come from the ``judge`` section of the config, overridable with the environment variables
    UNINAVBENCH_JUDGE_BASE_URL, UNINAVBENCH_JUDGE_MODEL and UNINAVBENCH_JUDGE_API_KEY. The API key is
    read from the environment variable named by ``api_key_env``.
    """

    def __init__(self, base_url=None, model=None, api_key_env="UNINAVBENCH_JUDGE_API_KEY", extra_body=None):
        from openai import OpenAI
        self.model = os.environ.get("UNINAVBENCH_JUDGE_MODEL", model or "default")
        env_body = os.environ.get("UNINAVBENCH_JUDGE_EXTRA_BODY")  # JSON, e.g. {"enable_thinking": false}
        self.extra_body = json.loads(env_body) if env_body else (extra_body or {})
        base_url = os.environ.get("UNINAVBENCH_JUDGE_BASE_URL", base_url)
        # configured lazily so that navigation-only evaluation needs no judge
        self.client = None if not base_url else OpenAI(
            base_url=base_url,
            api_key=os.environ.get(api_key_env) or os.environ.get("UNINAVBENCH_JUDGE_API_KEY") or "EMPTY",
        )

    def score(self, question, gt_answer, answer):
        """Returns an integer score in [1, 5]; unparsable judge output counts as 1."""
        if self.client is None:
            raise RuntimeError("Active-EQA needs an LLM judge: set UNINAVBENCH_JUDGE_BASE_URL, UNINAVBENCH_JUDGE_MODEL "
                               "and UNINAVBENCH_JUDGE_API_KEY (any OpenAI-compatible chat API), or UNINAVBENCH_JUDGE=exact")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Question: {question}\nAnswer: {gt_answer}\nResponse: {answer}"}
            ],
            temperature=0.0,
            max_tokens=1024,
            extra_body=self.extra_body or None,
        )
        content = response.choices[0].message.content or ""
        # reasoning models may emit <think>...</think> before the score
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
        match = re.search(r"[1-5]", content)
        return int(match.group()) if match else 1

    # name used by the original evaluator
    def scoring_eqa(self, question, gt_answer, answer):
        return self.score(question, gt_answer, answer)


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return sorted(_normalize(v) for v in value)
    if isinstance(value, bool):
        return str(value).lower()
    return re.sub(r"[^a-z0-9 ]", "", str(value).lower()).strip()


class ExactMatchJudge:
    """5 when the response equals the ground truth after normalisation, else 1. For offline tests only."""

    def score(self, question, gt_answer, answer):
        if isinstance(answer, str) and not isinstance(gt_answer, str):
            try:
                answer = json.loads(answer)
            except (ValueError, TypeError):
                pass
        return 5 if _normalize(gt_answer) == _normalize(answer) else 1

    scoring_eqa = score


def create_judge(config):
    judge = dict(config.get("judge") or {})
    kind = os.environ.get("UNINAVBENCH_JUDGE", judge.pop("type", "llm"))
    if kind == "exact":
        return ExactMatchJudge()
    if kind != "llm":
        raise ValueError(f"Unknown judge type: {kind}")
    return LLMJudge(**{k: v for k, v in judge.items() if k in ("base_url", "model", "api_key_env", "extra_body")})
