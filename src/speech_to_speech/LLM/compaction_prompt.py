"""Prompt template and factory for the conversation compaction (history summarization) function.

Compaction reduces an unbounded conversation history to a tight user/assistant
summary pair, letting the pipeline continue indefinitely without running out of
context window.  The factory :func:`build_compactor` returns a :data:`CompactFn`
compatible with :meth:`~speech_to_speech.LLM.chat.Chat.trim_if_needed`.

:data:`CompactGenerateFn` is the backend-agnostic generation interface:
``(system_prompt: str, user_prompt: str) -> response_text: str``.
Each handler wraps its own model into this interface and passes it to
:func:`build_compactor`.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from speech_to_speech.LLM.chat import CompactFn, CompactionResult

logger = logging.getLogger(__name__)

# Callable[[system_prompt, user_prompt], response_text]
CompactGenerateFn = Callable[[str, str], str]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPACTION_SYSTEM_PROMPT = """\
You are a conversation memory compressor for a real-time voice AI assistant.

Your task: read a multi-turn conversation and produce a dense summary so the
assistant can continue naturally, as if it remembers everything that was said.

Output a single JSON object with exactly two string fields:
  "user_summary"      — 1–5 sentences capturing what the user has been asking
                        about, any preferences or constraints they have stated,
                        and where the conversation stands from their perspective.
  "assistant_summary" — 1–5 sentences capturing what the assistant has
                        explained, decided, or done (including tool calls and
                        their results), plus any open questions or commitments.

Rules:
- Be information-dense: preserve names, numbers, file paths, error messages, and
  other specifics that would be needed to continue the conversation correctly.
- Omit small-talk and filler that carries no forward context.
- Write in third person, past tense
  (e.g. "The user asked about…", "The assistant provided…").
- Emit only the JSON object — no markdown, no code fences, no extra keys.\
"""

COMPACTION_USER_TEMPLATE = """\
Summarize the following conversation.  Return only the JSON object.

--- CONVERSATION START ---
{conversation}
--- CONVERSATION END ---\
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_transcript(snapshot: list[Any]) -> str:
    """Render a ResponseInputParam snapshot as a readable plain-text transcript."""
    lines: list[str] = []
    for item in snapshot:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "message")
        role: str = item.get("role", "")

        if role == "system":
            continue

        if item_type == "function_call":
            name = item.get("name", "")
            args = item.get("arguments", "")
            lines.append(f"[Tool call: {name}({args})]")
            continue

        if item_type == "function_call_output":
            out = item.get("output", "")
            lines.append(f"[Tool result: {out}]")
            continue

        # Regular user / assistant message
        content = item.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") in ("input_text", "output_text")
            ).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            continue

        if text:
            label = role.capitalize() if role else "Unknown"
            lines.append(f"{label}: {text}")

    return "\n\n".join(lines)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text*, stripping markdown code fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = _JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"No JSON object found in compaction response: {text!r}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_compactor(generate_fn: CompactGenerateFn) -> CompactFn:
    """Return a :data:`CompactFn` that summarizes history using *generate_fn*.

    *generate_fn* is the only model-specific dependency: it receives
    ``(system_prompt, user_prompt)`` and returns the model's text response.
    Both :class:`~speech_to_speech.LLM.openai_api_language_model.OpenApiModelHandler`
    and :class:`~speech_to_speech.LLM.language_model.BaseLanguageModelHandler`
    subclasses expose a ``_build_compaction_generate_fn()`` method that wraps
    their respective backend into this interface.

    The returned callable is safe to call from a background thread.

    Args:
        generate_fn: Backend-agnostic text generation callable.

    Returns:
        A callable ``(snapshot: ResponseInputParam) -> CompactionResult``.
    """

    def compact(snapshot: list[Any]) -> CompactionResult:
        transcript = _render_transcript(snapshot)
        if not transcript.strip():
            logger.warning("Compaction called with an empty transcript; returning empty summaries")
            return CompactionResult(user_summary="", assistant_summary="")

        user_content = COMPACTION_USER_TEMPLATE.format(conversation=transcript)
        raw_text = generate_fn(COMPACTION_SYSTEM_PROMPT, user_content)

        data = _extract_json(raw_text)
        user_summary = str(data.get("user_summary", "")).strip()
        assistant_summary = str(data.get("assistant_summary", "")).strip()

        if not user_summary or not assistant_summary:
            raise ValueError(f"Compaction response missing required fields. Got: {data!r}")

        logger.debug(
            "Compaction complete. user=%d chars  assistant=%d chars",
            len(user_summary),
            len(assistant_summary),
        )
        return CompactionResult(user_summary=user_summary, assistant_summary=assistant_summary)

    return compact
