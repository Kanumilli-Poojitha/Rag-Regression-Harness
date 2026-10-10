"""LLM-as-a-judge. Pinned model, temperature 0, strict binary output parsed in code."""
import json
from typing import List, Tuple

from . import settings
from .llm import chat


class JudgeError(RuntimeError):
    pass


def parse_binary(text: str) -> int:
    t = text.strip()
    if t in ("0", "1"):
        return int(t)
    raise JudgeError(f"judge returned non-binary output: {text!r}")


class Judge:
    def __init__(self, model: str, prompts_path=settings.PROMPTS_PATH):
        self.model = model
        with open(prompts_path, encoding="utf-8") as f:
            p = json.load(f)
        self.correctness_prompt = p["correctness_prompt"]
        self.groundedness_prompt = p["groundedness_prompt"]

    def _score(self, system: str, user: str, retries: int = 2) -> Tuple[int, str]:
        last = ""
        for _ in range(retries + 1):
            last = chat(self.model, [{"role": "system", "content": system},
                                     {"role": "user", "content": user}], max_tokens=3, temperature=0.0)
            try:
                return parse_binary(last), last
            except JudgeError:
                continue
        raise JudgeError(f"judge failed to return 0/1 after retries; last output {last!r}")

    def correctness(self, question: str, gt: str, answer: str) -> Tuple[int, str]:
        user = f"QUESTION: {question}\nGROUND TRUTH ANSWER: {gt}\nGENERATED ANSWER: {answer}\nSCORE:"
        return self._score(self.correctness_prompt, user)

    def groundedness(self, contexts: List[str], answer: str) -> Tuple[int, str]:
        user = "CONTEXT:\n" + "\n".join(contexts) + f"\nGENERATED ANSWER: {answer}\nSCORE:"
        return self._score(self.groundedness_prompt, user)
