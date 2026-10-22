from __future__ import annotations

from collections.abc import Sequence
from queue import Empty
from typing import Any

import nltk

# The pinned HF revision checks this tagger under ``tokenizers/`` even though
# NLTK installs it under ``taggers/``. Redirect only that lookup while HF's
# pipeline module initializes so an installed asset does not trigger a network
# download on every launch.
_nltk_find = nltk.data.find


def _find_hf_nltk_asset(resource_name: str, *args: Any, **kwargs: Any) -> Any:
    if resource_name == "tokenizers/averaged_perceptron_tagger_eng":
        resource_name = "taggers/averaged_perceptron_tagger_eng"
    return _nltk_find(resource_name, *args, **kwargs)


nltk.data.find = _find_hf_nltk_asset
try:
    from speech_to_speech.api.openai_realtime.audio_client import run_realtime_audio_client
    from speech_to_speech.cli import parse_command, parse_talk_arguments
finally:
    nltk.data.find = _nltk_find

from .backend import register_convfill_backend  # noqa: E402
from .delivery import DeliveryLedgerQueue  # noqa: E402
from .handler import ConvFillRealtimeHandler  # noqa: E402


def _attach_audio_state(manager: Any) -> Any:
    """Swap each unit's audio output queue for a ConvFill delivery ledger.

    Every assistant text event and audio chunk the realtime send loop dispatches
    is dequeued from this queue, so subclassing it is the only place ConvFill can
    observe what the browser was actually sent. ``build_pipeline`` returns before
    ``ThreadManager.start`` launches any handler thread, so the swap happens while
    the queue is quiescent.
    """
    realtime_server = next((handler for handler in manager.handlers if hasattr(handler, "pool")), None)
    if realtime_server is None:
        raise RuntimeError("Could not locate Hugging Face RealtimeServer in the pipeline manager.")
    for unit in realtime_server.pool:
        adapter = next(
            (handler for handler in unit.handlers if isinstance(handler, ConvFillRealtimeHandler)),
            None,
        )
        if adapter is None:
            raise RuntimeError(f"Pipeline {unit.index} has no ConvFillRealtimeHandler.")

        previous_queue = unit.output_queue
        # The TTS handler also writes to its own ``queue_out`` reference
        # directly, so both bindings have to move together.
        producers = [handler for handler in unit.handlers if getattr(handler, "queue_out", None) is previous_queue]
        if not producers:
            raise RuntimeError(f"Pipeline {unit.index} has no handler writing to the audio output queue.")

        ledger = DeliveryLedgerQueue()
        while True:
            try:
                ledger.put_nowait(previous_queue.get_nowait())
            except Empty:
                break

        unit.output_queue = ledger
        for handler in producers:
            handler.queue_out = ledger
        adapter.set_delivery_ledger(ledger)
    return manager


def main(argv: Sequence[str] | None = None) -> None:
    # Register only in this opt-in entry point, before the pipeline creates its
    # argument dataclasses. Importing delivery helpers must not change defaults.
    register_convfill_backend()
    nltk.data.find = _find_hf_nltk_asset
    try:
        from speech_to_speech import s2s_pipeline
    finally:
        nltk.data.find = _nltk_find
    command, command_args = parse_command(argv)
    if command == "talk":
        run_realtime_audio_client(parse_talk_arguments(command_args))
        return

    # The pinned upstream builder owns the full VAD/STT/TTS/realtime stack. A
    # narrow wrapper swaps in the delivery ledger ConvFill needs to schedule
    # <sil> fillers and to keep an interrupted turn in history.
    original_build_pipeline = s2s_pipeline.build_pipeline

    def build_pipeline_with_convfill(*args: Any, **kwargs: Any) -> Any:
        return _attach_audio_state(original_build_pipeline(*args, **kwargs))

    s2s_pipeline.build_pipeline = build_pipeline_with_convfill
    s2s_pipeline.run_pipeline_command(command, command_args)


if __name__ == "__main__":
    main()
