"""Immutable history proposals from a model's private conversation snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from openai.types.realtime.conversation_item import RealtimeConversationItemFunctionCall

from speech_to_speech.LLM.chat import Chat, CompactFn, SupportedItem


@dataclass(frozen=True)
class ResponseHistory:
    """Cumulative write-back; serialized items cannot change after emission.

    Tools carry the prefix needed before exposing their call. The terminal
    carries the full generation and cleanup policy. Only RealtimeService may
    apply either to the shared chat. The compactor computes a summary from a
    snapshot; it is not a callback that writes shared history.
    """

    chat_id: int
    items: tuple[tuple[type[SupportedItem], str], ...]
    after_item_id: str | None
    complete: bool = False
    input_item_id: str | None = None
    consumed_image_ids: frozenset[str] = frozenset()
    item_order: tuple[str, ...] | None = None
    audio_history_turns: int | None = None
    compactor: CompactFn | None = None

    @classmethod
    def capture(
        cls,
        chat: Chat,
        items: list[SupportedItem],
        *,
        after_item_id: str | None,
        complete: bool = False,
        input_item_id: str | None = None,
        consumed_image_ids: set[str] | None = None,
        item_order: list[str] | None = None,
        audio_history_turns: int | None = None,
        compactor: CompactFn | None = None,
    ) -> ResponseHistory:
        # Assign/validate IDs in a private scratch chat, never in the live chat.
        scratch = Chat(0)
        encoded = []
        for item in items:
            copied = item.model_copy(deep=True)
            if isinstance(copied, RealtimeConversationItemFunctionCall):
                scratch.add_ordered_function_call(copied)
            else:
                scratch.add_item(copied)
            encoded.append((type(copied), copied.model_dump_json(exclude_unset=True)))
            # Reuse those stable IDs in the next cumulative proposal.
            item.id = copied.id
        return cls(
            chat_id=id(chat),
            items=tuple(encoded),
            after_item_id=after_item_id,
            complete=complete,
            input_item_id=input_item_id,
            consumed_image_ids=frozenset(consumed_image_ids or ()),
            item_order=tuple(item_order) if item_order is not None else None,
            audio_history_turns=audio_history_turns,
            compactor=compactor,
        )

    def decode_items(self, start: int = 0) -> list[SupportedItem]:
        return [kind.model_validate_json(value) for kind, value in self.items[start:]]
