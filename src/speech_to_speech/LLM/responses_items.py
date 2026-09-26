from __future__ import annotations

import logging
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem

from speech_to_speech.LLM.chat import ReasoningRecord
from speech_to_speech.utils.utils import _generate_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssembledAssistant:
    content: list
    leading_reasoning: tuple[ReasoningRecord, ...]


@dataclass(frozen=True)
class AssembledToolCall:
    item: ResponseFunctionToolCall
    leading_reasoning: tuple[ReasoningRecord, ...]
    replay_exact: bool


def reasoning_from_wire(item: Any) -> ReasoningRecord:
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            payload = deepcopy(dump(exclude_unset=True, mode="json"))
        except TypeError:
            payload = deepcopy(dump())
    elif isinstance(item, dict):
        payload = deepcopy(item)
    else:
        payload = {}
        for key in ("id", "type", "summary", "encrypted_content", "status", "content"):
            if hasattr(item, key):
                value = getattr(item, key)
                if value is not None:
                    payload[key] = deepcopy(value)
    item_id = payload.get("id") or getattr(item, "id", None) or ""
    if not item_id:
        raise ValueError("reasoning item requires a non-empty id")
    payload["id"] = item_id
    if payload.get("type") is None:
        payload["type"] = "reasoning"
    return ReasoningRecord(id=str(item_id), payload=payload)


def adopt_provider_ids(item: ResponseFunctionToolCall) -> bool:
    item_id = item.id or ""
    call_id = item.call_id or ""
    call_ok = call_id.startswith("call_")
    id_ok = not item_id or item_id.startswith("fc_")
    if call_ok and id_ok:
        if not item_id:
            item.id = _generate_id("fc")
        return True
    item.id = _generate_id("fc")
    item.call_id = _generate_id("call")
    return False


def _is_reasoning_item(item: Any) -> bool:
    return isinstance(item, ResponseReasoningItem) or getattr(item, "type", None) == "reasoning"


def _assistant_content(content: Any) -> list[AssistantContent]:
    return [
        AssistantContent(type="output_text", text=c.text if c.type == "output_text" else c.refusal) for c in content
    ]


class ResponsesSegmentAssembler:
    """Buffer reasoning until the next function call or assistant message."""

    def __init__(self, *, capture_reasoning: bool = True) -> None:
        self.capture_reasoning = capture_reasoning
        self._pending: list[ReasoningRecord] = []

    def _take(self) -> tuple[ReasoningRecord, ...]:
        leading = tuple(self._pending)
        self._pending.clear()
        return leading

    def feed_item(self, item: Any) -> Iterator[AssembledAssistant | AssembledToolCall]:
        if _is_reasoning_item(item):
            if self.capture_reasoning:
                self._pending.append(reasoning_from_wire(item))
            return
        if isinstance(item, ResponseFunctionToolCall):
            replay_exact = adopt_provider_ids(item)
            if replay_exact:
                leading = self._take()
            else:
                if self._pending:
                    logger.debug(
                        "Dropping %d reasoning item(s); function call ids were reissued",
                        len(self._pending),
                    )
                self._pending.clear()
                leading = ()
            yield AssembledToolCall(item=item, leading_reasoning=leading, replay_exact=replay_exact)
            return
        if isinstance(item, ResponseOutputMessage):
            yield AssembledAssistant(content=_assistant_content(item.content), leading_reasoning=self._take())
            return
        logger.warning(f"Not supported message type: {getattr(item, 'type', None)}")

    def finish(self) -> Iterator[AssembledAssistant | AssembledToolCall]:
        if self._pending:
            logger.debug(
                "Dropping trailing reasoning with no host (%s)",
                [record.id for record in self._pending],
            )
            self._pending.clear()
        return iter(())
