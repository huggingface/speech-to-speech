import logging
from typing import Any

logger = logging.getLogger(__name__)


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.init_chat_message: dict[str, Any] | None = None
        # ``size`` is the number of user turns to keep.  When exceeded the
        # oldest complete turn (everything up to the next ``role: "user"``)
        # is evicted.
        self.buffer: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}

    def append(self, item: dict[str, Any]) -> None:
        self.buffer.append(item)
        self._track_tool_calls(item)
        while self._count_user_turns() > self.size:
            self._evict_oldest_turn()

    def _count_user_turns(self) -> int:
        return sum(1 for item in self.buffer if item.get("role") == "user")

    def _evict_oldest_turn(self) -> None:
        """Remove items from the front until the next ``role: "user"`` boundary."""
        if not self.buffer:
            return
        self.buffer.pop(0)
        while self.buffer and self.buffer[0].get("role") != "user":
            self.buffer.pop(0)

    def _track_tool_calls(self, item: dict[str, Any]) -> None:
        """Register any tool call IDs found in *item* for pending tracking.

        Handles both the Responses API format (``type: "function_call"``) and
        the Chat Completions / transformers format (``role: "assistant"`` with
        a ``tool_calls`` list).
        """
        if item.get("type") == "function_call" and item.get("call_id"):
            self._pending_tool_calls[item["call_id"]] = item
        elif item.get("role") == "assistant" and item.get("tool_calls"):
            for tc in item["tool_calls"]:
                tc_id = tc.get("id", "")
                if tc_id:
                    self._pending_tool_calls[tc_id] = item

    def _has_call_id_in_buffer(self, call_id: str) -> bool:
        """Check whether *call_id* exists in the buffer (either format)."""
        for entry in self.buffer:
            if entry.get("type") == "function_call" and entry.get("call_id") == call_id:
                return True
            if entry.get("role") == "assistant" and entry.get("tool_calls"):
                for tc in entry["tool_calls"]:
                    if tc.get("id") == call_id:
                        return True
        return False

    def append_tool_output(self, call_id: str, output_item: dict[str, Any]) -> str | None:
        """Append a ``function_call_output``, re-injecting its ``function_call`` if evicted.

        Returns ``None`` on success or an error message if *call_id* is unknown.
        """
        if self._has_call_id_in_buffer(call_id):
            self._pending_tool_calls.pop(call_id, None)
            self.append(output_item)
            return None

        if call_id in self._pending_tool_calls:
            logger.info("Re-injecting evicted function_call for call_id=%s", call_id)
            self.append(self._pending_tool_calls.pop(call_id))
            self.append(output_item)
            self._pending_tool_calls.pop(call_id, None)
            return None

        return f"No function_call with call_id '{call_id}' found in conversation history."

    def init_chat(self, init_chat_message: dict[str, Any]) -> None:
        self.init_chat_message = init_chat_message

    def to_list(self) -> list[dict[str, Any]]:
        if self.init_chat_message:
            return [self.init_chat_message] + self.buffer
        else:
            return self.buffer

    def copy(self) -> "Chat":
        """Return a shallow snapshot safe for concurrent read access."""
        clone = Chat(self.size)
        clone.init_chat_message = self.init_chat_message
        clone.buffer = list(self.buffer)
        clone._pending_tool_calls = dict(self._pending_tool_calls)
        return clone

    def reset(self) -> None:
        self.buffer = []
        self.init_chat_message = None
        self._pending_tool_calls = {}

    def strip_images(self) -> None:
        """Remove all image content parts from every message in the buffer.

        Called after appending the assistant response so images don't persist
        across turns.  Handles both Realtime-style ``input_image`` and
        Chat Completions-style ``image_url`` types.
        """
        _IMAGE_TYPES = {"input_image", "image_url"}
        for msg in self.buffer:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            text_parts = [
                p
                for p in content
                if (p.get("type") if isinstance(p, dict) else getattr(p, "type", None)) not in _IMAGE_TYPES
            ]
            msg["content"] = text_parts
