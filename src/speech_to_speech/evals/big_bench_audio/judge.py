"""Optional LLM judge for replies the rule-based extractor cannot read.

The judge only *extracts* an answer token; whether that token is correct is still
decided by string comparison against the dataset. A judge that also ruled on
correctness would drift between runs, which is exactly what a comparison harness
cannot afford.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from speech_to_speech.evals.big_bench_audio.report import ItemResult
from speech_to_speech.evals.big_bench_audio.scoring import CATEGORY_SPECS, normalize_official

logger = logging.getLogger(__name__)

_FORMATS = {
    "formal_fallacies": "the single word `valid` or `invalid`",
    "navigate": "the single word `Yes` or `No`",
    "web_of_lies": "the single word `Yes` or `No`",
    "object_counting": "a single whole number, digits only",
}

_SYSTEM_PROMPT = (
    "You extract the final answer from a transcript of someone answering a reasoning question "
    "out loud. Reply with the answer only: no punctuation, no explanation, no restating the "
    "question. If the transcript never commits to an answer, reply with exactly UNPARSED."
)


@dataclass
class JudgeConfig:
    """Where to reach the judge model."""

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    concurrency: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url}


def _user_prompt(result: ItemResult) -> str:
    fmt = _FORMATS.get(result.category, "the answer only")
    return f"Question category: {result.category}\nAnswer format: {fmt}\n\nTranscript:\n{result.reply}\n\nFinal answer:"


def _normalize_reply(category: str, text: str) -> Optional[str]:
    cleaned = text.strip().strip(".").strip()
    if not cleaned or cleaned.upper() == "UNPARSED":
        return None
    spec = CATEGORY_SPECS.get(category)
    if spec is not None and spec.kind == "integer":
        digits = "".join(char for char in cleaned if char.isdigit())
        return digits or None
    return cleaned.lower()


async def judge_results(results: list[ItemResult], config: JudgeConfig) -> None:
    """Fill in ``judge_extracted`` / ``judge_correct`` on every answered item, in place."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key or "not-needed")
    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    async def judge_one(result: ItemResult) -> None:
        if not result.reply.strip():
            return
        async with semaphore:
            try:
                completion = await client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": _user_prompt(result)},
                    ],
                    temperature=0,
                    max_tokens=16,
                    timeout=config.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001 — a judge outage must not void the run
                logger.warning("judge failed for item %s: %s", result.id, exc)
                return

        raw = (completion.choices[0].message.content or "") if completion.choices else ""
        extracted = _normalize_reply(result.category, raw)
        if extracted is None:
            return
        result.judge_extracted = extracted
        result.judge_correct = extracted == normalize_official(result.category, result.official_answer)

    await asyncio.gather(*(judge_one(result) for result in results))
