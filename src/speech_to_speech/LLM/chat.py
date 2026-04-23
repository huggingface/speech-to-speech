from typing import Any


class Chat:
    """
    Handles the chat using to avoid OOM issues.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.init_chat_message: dict[str, Any] | None = None
        # maxlen is necessary pair, since a each new step we add an prompt and assitant answer
        self.buffer: list[dict[str, Any]] = []

    def append(self, item: dict[str, Any]) -> None:
        self.buffer.append(item)
        if len(self.buffer) == 2 * (self.size + 1):
            self.buffer.pop(0)
            self.buffer.pop(0)

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
        return clone

    def reset(self) -> None:
        self.buffer = []
        self.init_chat_message = None

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
