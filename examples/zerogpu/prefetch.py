"""Observe native confirmed speech events without changing turn handling."""

import logging
from threading import Lock, Thread

from speech_to_speech.pipeline.events import SpeechStartedEvent


class PrefetchQueue:
    """Forward the original queue and asynchronously request a GPU on speech.

    This is attached only to VAD's output-event queue. The original queue is
    still consumed by RealtimeService; no event is removed or transformed.
    """

    def __init__(self, queue, prefetch):
        self.queue = queue
        self.prefetch = prefetch
        self._in_flight = Lock()

    def put(self, item, *args, **kwargs):
        self.queue.put(item, *args, **kwargs)
        if isinstance(item, SpeechStartedEvent) and self._in_flight.acquire(blocking=False):
            Thread(target=self._request, daemon=True).start()

    def _request(self):
        try:
            self.prefetch()
        except Exception:
            # Prefetch is optional. Ordinary inference reports allocation
            # errors through the existing response/error path when needed.
            logging.getLogger(__name__).exception("Early GPU allocation failed")
        finally:
            self._in_flight.release()

    def put_nowait(self, item):
        return self.put(item, block=False)

    def __getattr__(self, name):
        return getattr(self.queue, name)


def enable_speech_prefetch(unit, prefetch):
    """Install after native pipeline construction, before handler threads start."""
    vad = unit.handlers[0]
    vad.text_output_queue = PrefetchQueue(vad.text_output_queue, prefetch)
