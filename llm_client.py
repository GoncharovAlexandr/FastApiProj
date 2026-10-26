"""Клиент внешнего OpenAI-compatible LLM API."""
import time

import httpx

from config import LLM_MODEL


class LLMServiceError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


async def chat_completion(
    client: httpx.AsyncClient,
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 300,
) -> dict:
    start = time.perf_counter()
    try:
        response = await client.post(
            "chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты помощник сервиса агроскоринга. Отвечай по-русски, "
                            "кратко и понятно. Не выдумывай факты о хозяйстве."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    except httpx.TimeoutException as exc:
        raise LLMServiceError("LLM provider timeout", 504) from exc
    except httpx.RequestError as exc:
        raise LLMServiceError("LLM provider is unavailable", 503) from exc

    if response.status_code == 429:
        raise LLMServiceError("LLM provider is temporarily unavailable", 503)
    if response.status_code < 200 or response.status_code >= 300:
        raise LLMServiceError("LLM provider returned an error", 502)

    try:
        payload = response.json()
        choice = payload["choices"][0]
        answer = choice["message"]["content"]
        if not isinstance(answer, str):
            raise TypeError("LLM answer is not text")
        returned_model = str(payload.get("model") or model)
        finish_reason = choice.get("finish_reason")
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMServiceError("LLM provider returned an invalid response", 502) from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "requested_model": model,
        "returned_model": returned_model,
        "answer": answer,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
    }
