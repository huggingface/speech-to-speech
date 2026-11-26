"""JSON chat calls to Hugging Face Inference Providers (OpenAI-compatible router).

Each call walks an ordered list of models and returns the first reply that
parses as a JSON object, so one provider hiccup never stalls a quiz turn.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")


def _models(env: str, default: str) -> list[str]:
    raw = os.environ.get(env, "").strip() or default
    return [m.strip() for m in raw.split(",") if m.strip()]


# Grading runs on every spoken answer, so it favours speed, and its clues
# must not give answers away: in a leak test over 12 guided dialogues
# gpt-oss-120b on Cerebras leaked once at ~0.5 s per turn, against 4 to 5
# leaks for Qwen3.8-27B, GLM-5.3 and GLM-5.3-Flash.
GRADER_MODELS = _models(
    "GRADER_MODELS",
    "openai/gpt-oss-120b:cerebras,openai/gpt-oss-120b:groq,Qwen/Qwen3.8-27B:cerebras,zai-org/GLM-5.3-Flash",
)
# The grader can run on the speech backend's own model instead (any
# OpenAI-compatible endpoint, e.g. the deployment's Gemma 4 31B), with the
# hosted chain above as fallback. On the same tests Gemma 4 31B accepted as
# many brief correct answers, leaked no answers, and took ~1.3 s per turn.
Route = tuple[str, str, str | None]  # base URL, model, reasoning effort


def _grader_routes() -> list[str | Route]:
    base = os.environ.get("GRADER_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("GRADER_MODEL", "").strip()
    primary: list[str | Route] = []
    if base and model:
        primary.append((base, model, os.environ.get("GRADER_REASONING_EFFORT", "none").strip() or None))
    return primary + list(GRADER_MODELS)


GRADER_ROUTES = _grader_routes()
# Question writing runs once per topic, so it favours quality.
WRITER_MODELS = _models(
    "WRITER_MODELS",
    "zai-org/GLM-5.3,MiniMaxAI/MiniMax-M3.1-Flash,openai/gpt-oss-120b:cerebras",
)


class LLMError(RuntimeError):
    pass


def api_token() -> str:
    return (os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN") or "").strip()


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (tolerates fences and prose)."""
    text = _FENCE.sub("", (text or "").strip())
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in model reply")


_client: httpx.AsyncClient | None = None


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
    return _client


async def chat_json(
    messages: list[dict[str, str]],
    models: list[str] | list[str | Route],
    *,
    max_tokens: int,
    temperature: float = 0.4,
    timeout: float = 30.0,
    reasoning_effort: str | None = os.environ.get("LLM_REASONING_EFFORT", "low") or None,
) -> tuple[dict[str, Any], str]:
    """Return ``(parsed_json, model)`` from the first model that answers well.

    Reasoning models otherwise spend thousands of hidden tokens before the
    JSON, so ``reasoning_effort`` defaults to low. A provider that rejects the
    field (HTTP 400) is retried once without it.
    """
    token = api_token()
    if not token:
        raise LLMError("No HF_TOKEN configured for the coach's language model.")
    errors: list[str] = []
    for entry in models:
        base_url, model, effort = entry if isinstance(entry, tuple) else (BASE_URL, entry, reasoning_effort)
        key = (os.environ.get("GRADER_API_KEY") or token) if isinstance(entry, tuple) else token
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if effort:
            body["reasoning_effort"] = effort
        for attempt in range(2):
            started = time.perf_counter()
            try:
                resp = await _http().post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
                data = resp.json()
                choices = data.get("choices") if isinstance(data, dict) else None
                if resp.status_code == 400 and "reasoning_effort" in body:
                    body.pop("reasoning_effort")
                    continue
                if resp.status_code != 200 or not choices:
                    raise LLMError(f"HTTP {resp.status_code}: {str(data)[:200]}")
                content = choices[0].get("message", {}).get("content") or ""
                parsed = extract_json(content)
                logger.info("llm %s answered in %.2fs", model, time.perf_counter() - started)
                return parsed, model
            except (httpx.HTTPError, LLMError, ValueError) as exc:
                logger.warning("llm %s failed after %.2fs: %s", model, time.perf_counter() - started, exc)
                errors.append(f"{model}: {exc}")
                break
    raise LLMError("All models failed: " + " | ".join(errors))
