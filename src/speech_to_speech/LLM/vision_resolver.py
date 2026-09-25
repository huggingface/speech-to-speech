from __future__ import annotations

import logging
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_VISION_SYSTEM_PROMPT = (
    "Describe what is relevant in this image accurately and concisely in 2-4 sentences."
)


class VisionResolver:
    """Stateless vision oracle using an OpenAI-compatible endpoint.

    Converts (image_url, question) into a text observation so the main conversation
    LLM can remain text-only.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 300,
        timeout_s: float = 10.0,
        system_prompt: str = DEFAULT_VISION_SYSTEM_PROMPT,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_s = float(timeout_s)
        self.system_prompt = system_prompt
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def resolve(
        self,
        image_urls: str | list[str],
        question: str,
        cancel_scope: Any = None,
    ) -> str:
        """Resolve image(s) and question to a concise text description.

        On error or timeout, returns a fallback message so conversation can continue.
        """
        if cancel_scope is not None and getattr(cancel_scope, "is_stale", lambda _: False)(
            getattr(cancel_scope, "generation", None)
        ):
            logger.info("VisionResolver: skipping resolution for stale generation")
            return "image resolution cancelled"

        urls = [image_urls] if isinstance(image_urls, str) else image_urls
        user_content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": url}} for url in urls
        ]
        user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        t0 = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=self.max_tokens,
                timeout=self.timeout_s,
            )
            latency = time.perf_counter() - t0
            resolved_text = (response.choices[0].message.content or "").strip()
            usage = response.usage
            if usage:
                logger.info(
                    "VisionResolver finished in %.3fs (prompt_tokens=%s, completion_tokens=%s, total_tokens=%s)",
                    latency,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                )
            else:
                logger.info("VisionResolver finished in %.3fs", latency)
            return resolved_text or "image analyzed with no additional details"
        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.warning("VisionResolver call failed after %.3fs: %s", latency, exc)
            return "image could not be analyzed in time"
