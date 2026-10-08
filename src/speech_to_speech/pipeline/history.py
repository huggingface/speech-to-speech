"""History proposals that a model emits instead of writing shared chat."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from speech_to_speech.LLM.chat import Chat, CompactFn, SupportedItem


@dataclass(frozen=True)
class ResponseHistory:
    """Cumulative write-back for one response, copied when the model emits it.

    A tool call carries the items needed before the client sees the call; the
    terminal carries the whole response plus its cleanup policy. Only
    RealtimeService applies either one to ``chat``.
    """

    chat: Chat = field(repr=False, compare=False)
    items: tuple[SupportedItem, ...]
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
        items: Iterable[SupportedItem],
        *,
        after_item_id: str | None,
        complete: bool = False,
        input_item_id: str | None = None,
        consumed_image_ids: Iterable[str] = (),
        item_order: Iterable[str] | None = None,
        audio_history_turns: int | None = None,
        compactor: CompactFn | None = None,
    ) -> ResponseHistory:
        return cls(
            chat=chat,
            items=tuple(item.model_copy(deep=True) for item in items),
            after_item_id=after_item_id,
            complete=complete,
            input_item_id=input_item_id,
            consumed_image_ids=frozenset(consumed_image_ids),
            item_order=tuple(item_order) if item_order is not None else None,
            audio_history_turns=audio_history_turns,
            compactor=compactor,
        )

    def write(self, response_key: str, start: int = 0, after_item_id: str | None = None) -> list[SupportedItem]:
        """Write ``items[start:]`` as reversible output of *response_key*."""
        return self.chat.add_provisional_generation_items(
            response_key,
            [item.model_copy(deep=True) for item in self.items[start:]],
            after_item_id=after_item_id or self.after_item_id,
            committed_item_ids={self.input_item_id} if self.complete and self.input_item_id else None,
        )

    def clean_up(self) -> None:
        """Apply the terminal cleanup policy, undoing all of it on failure."""
        chat = self.chat
        snapshot = chat.snapshot_history_cleanup()
        try:
            if self.item_order is not None:
                chat.order_response_items(self.item_order)
            chat.strip_images(set(self.consumed_image_ids))
            if self.audio_history_turns is not None:
                chat.compact_audio_history(self.audio_history_turns)
            chat.trim_if_needed(self.compactor)
        except Exception:
            chat.restore_history_cleanup(snapshot)
            raise
