"""A pipeline output queue that records what leaves it.

The realtime send loop pulls every assistant text event and audio chunk off
``PipelineUnit.output_queue``. Observing each dequeue lets ConvFill estimate
when the listener would otherwise hear silence, so ``<sil>`` fillers follow
playback rather than the coarse ``response_playing`` event, which stays set for
a whole response.

A dequeue is not proof of dispatch: barge-in flushes and held or stale output
also leave through ``get()``. The ledger therefore drives filler timing only;
it is never treated as the text a listener heard.
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from typing import Any

from speech_to_speech.api.openai_realtime.service import PIPELINE_SAMPLE_RATE
from speech_to_speech.api.openai_realtime.websocket_router import _is_audio_done, _is_pipeline_end, _to_audio_bytes
from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import AudioOutput

logger = logging.getLogger(__name__)

# Output queue audio is pipeline-rate mono PCM16; the client rate is applied later.
_AUDIO_BYTES_PER_SECOND = PIPELINE_SAMPLE_RATE * 2


def audio_duration_s(item: Any) -> float:
    """Seconds of playable audio carried by *item*, or 0.0 if it carries none."""
    if isinstance(item, AudioOutput):
        if item.cleanup_only:
            return 0.0
    elif not isinstance(item, (bytes, bytearray)) and not hasattr(item, "tobytes"):
        return 0.0
    if _is_audio_done(item) or _is_pipeline_end(item):
        return 0.0
    try:
        raw = _to_audio_bytes(item)
    except (TypeError, AttributeError):
        return 0.0
    if not isinstance(raw, (bytes, bytearray)):
        return 0.0
    return len(raw) / _AUDIO_BYTES_PER_SECOND


def _event_texts(item: AssistantOutputEvent) -> list[str]:
    texts = [
        str(part.text)
        for part in getattr(item, "parts", [])
        if getattr(part, "type", None) == "text" and str(getattr(part, "text", "")).strip()
    ]
    if texts:
        return texts
    legacy = str(getattr(item, "text", "") or "")
    return [legacy] if legacy.strip() else []


class DeliveryLedgerQueue(Queue):
    """A pipeline output queue that ledgers everything taken from it."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        # Kept separate from ``self.mutex`` so ledger reads never contend with,
        # or deadlock against, the queue's own locking.
        self._ledger_lock = threading.Lock()
        self._delivered: dict[str, list[str]] = {}
        self._playhead = time.monotonic()

    # ``get_nowait`` funnels through ``get``, so this observes every dequeue.
    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        item = super().get(block, timeout)
        try:
            self._record(item)
        except Exception:  # never let bookkeeping break the transport
            logger.debug("Delivery ledger could not record %r", type(item), exc_info=True)
        return item

    def _record(self, item: Any) -> None:
        if isinstance(item, AssistantOutputEvent):
            texts = _event_texts(item)
            key = getattr(item, "response_key", None)
            if texts and key is not None:
                with self._ledger_lock:
                    self._delivered.setdefault(str(key), []).extend(texts)
            return

        duration = audio_duration_s(item)
        if duration <= 0.0:
            return
        now = time.monotonic()
        with self._ledger_lock:
            self._playhead = max(self._playhead, now) + duration

    def delivered_text(self, response_key: str | None) -> list[str]:
        """Assistant sentences taken from the queue for *response_key*."""
        if response_key is None:
            return []
        with self._ledger_lock:
            return list(self._delivered.get(str(response_key), []))

    def clear_response(self, response_key: str | None) -> None:
        if response_key is None:
            return
        with self._ledger_lock:
            self._delivered.pop(str(response_key), None)

    def audio_playing(self) -> bool:
        """Estimate whether dispatched audio may still be playing (no client acknowledgement)."""
        with self._ledger_lock:
            return time.monotonic() < self._playhead

    def reset_playback(self) -> None:
        """Forget queued playback time; the client discards its buffer on barge-in."""
        with self._ledger_lock:
            self._playhead = time.monotonic()

    def pending_audio(self) -> bool:
        """Whether any audio is still waiting to be dispatched."""
        with self.mutex:
            items = list(self.queue)
        return any(audio_duration_s(item) > 0.0 for item in items)
