"""Thin OpenAI wrapper with retry/backoff. Key and base URL come from the environment."""
import time
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
    return _client


def chat(model: str, messages: list, max_tokens: int, temperature: float = 0.0, attempts: int = 5) -> str:
    delay = 1.0
    for i in range(attempts):
        try:
            r = get_client().chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, seed=0,
            )
            return (r.choices[0].message.content or "").strip()
        except (RateLimitError, APIConnectionError, APIError):
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")
