"""A pipeline output queue that records what the transport dequeued for dispatch.

Hugging Face's realtime send loop pulls every assistant text event and every
audio chunk off ``PipelineUnit.output_queue`` with ``get_nowait()``. Subclassing
that queue and observing each dequeue gives ConvFill two facts it cannot get
anywhere else:

* which Talker sentences were handed to the client for a given response key, so
  an interrupted turn can retain a best-effort dispatched prefix, and
* how much audio has been dispatched, so ``<sil>`` filler scheduling can be
  driven by an estimated playback duration rather than Hugging Face's coarse
  ``response_playing`` event, which stays set for a whole response.

Barge-in flushes drain ``q.queue`` directly under ``q.mutex`` without calling
``get()``. That path stays intact here: only ``get`` is overridden, and the
underlying deque and mutex are untouched.
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from typing import Any

from speech_to_speech.pipeline.events import AssistantOutputEvent
from speech_to_speech.pipeline.messages import AudioOutput

try:  # Prefer upstream's own payload helpers so the accounting cannot drift.
    from speech_to_speech.api.openai_realtime.websocket_router import (
        _is_audio_done as _hf_is_audio_done,
    )
    from speech_to_speech.api.openai_realtime.websocket_router import (
        _is_pipeline_end as _hf_is_pipeline_end,
    )
    from speech_to_speech.api.openai_realtime.websocket_router import (
        _to_audio_bytes as _hf_to_audio_bytes,
    )
except ImportError:  # pragma: no cover - only if upstream renames its helpers
    _hf_is_audio_done = None
    _hf_is_pipeline_end = None
    _hf_to_audio_bytes = None

logger = logging.getLogger(__name__)

# Pipeline audio is 24 kHz mono signed 16-bit PCM.
AUDIO_SAMPLE_RATE_HZ = 24000
_AUDIO_BYTES_PER_SECOND = AUDIO_SAMPLE_RATE_HZ * 2

_SENTINEL_PAYLOADS = (b"__RESPONSE_DONE__", b"END")


def _payload(item: Any) -> Any:
    return item.audio if isinstance(item, AudioOutput) else item


def _is_sentinel(item: Any) -> bool:
    if _hf_is_audio_done is not None and _hf_is_pipeline_end is not None:
        return bool(_hf_is_audio_done(item) or _hf_is_pipeline_end(item))
    payload = _payload(item)
    return isinstance(payload, bytes) and payload in _SENTINEL_PAYLOADS


def audio_duration_s(item: Any) -> float:
    """Seconds of playable audio carried by *item*, or 0.0 if it carries none."""
    if isinstance(item, AudioOutput):
        if item.cleanup_only:
            return 0.0
    elif not isinstance(item, (bytes, bytearray)) and not hasattr(item, "tobytes"):
        return 0.0
    if _is_sentinel(item):
        return 0.0
    try:
        if _hf_to_audio_bytes is not None:
            raw = _hf_to_audio_bytes(item)
        else:  # pragma: no cover - fallback for a renamed upstream helper
            payload = _payload(item)
            raw = payload.tobytes() if hasattr(payload, "tobytes") else payload
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
    """A pipeline output queue that ledgers everything the send loop dequeues."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        # Kept separate from ``self.mutex`` so ledger reads never contend with,
        # or deadlock against, the queue's own locking.
        self._ledger_lock = threading.Lock()
        self._delivered: dict[str, list[str]] = {}
        self._playhead = time.monotonic()

    # ``get_nowait`` funnels through ``get``, and so does the send loop's batch
    # drain, so this is the single observation point for dispatched output.
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
        """Assistant sentences dispatched to the client for *response_key*."""
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
