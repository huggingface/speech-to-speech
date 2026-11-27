from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.turn_latency import TTSResponseTimer, TurnLatencyStore
from speech_to_speech.utils.mlx_lock import suspend_mlx_lock

logger = logging.getLogger(__name__)


class BaseTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Shared base for TTS handlers: speculative turn gating and response latency bookkeeping."""

    # Handlers assign these in setup() or are wired after construction.
    cancel_scope: CancelScope | None = None
    turn_latency_store: TurnLatencyStore | None = None
    speculative_turns: SpeculativeTurnTracker | None = None
    _print_assistant_text: bool = True
    _active_tts_input: TTSInput | None = None
    _active_timer: TTSResponseTimer | None = None

    def _on_end_of_response(self, end: EndOfResponse) -> None:
        """Reset per-response handler state when an EndOfResponse arrives."""

    def _synthesize_input(self, tts_input: TTSInput) -> Iterator[TTSOut]:
        """Yield synthesized audio blocks for one admitted ``TTSInput``."""
        raise NotImplementedError

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            self._on_end_of_response(tts_input)
            if self._gate_end_of_response(tts_input):
                yield AUDIO_RESPONSE_DONE
            return

        if not self._admit_input(tts_input):
            return

        self._active_tts_input = tts_input
        self._active_timer = None
        process_generation = tts_input.cancel_generation
        if process_generation is None and self.cancel_scope is not None:
            process_generation = self.cancel_scope.generation

        gen = self._synthesize_input(tts_input)
        try:
            first_audio = True
            for output in gen:
                if first_audio:
                    if not self._commit_first_audio(tts_input, generation=process_generation):
                        return
                    first_audio = False
                    timer = self._active_timer
                    e2e_s = timer.commit() if timer is not None else None
                    self._on_first_audio_committed(tts_input, timer, e2e_s)
                yield output
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
            self._active_tts_input = None
            self._active_timer = None

    def _gate_end_of_response(self, end: EndOfResponse) -> bool:
        """Wait for the speculative turn gate and say whether to forward the end marker.

        A superseded turn with a ``response_key`` still forwards the marker, flagged
        ``cleanup_only``, so downstream state is released. Without a key it is dropped.
        """
        tracker = self.speculative_turns
        if tracker and not tracker.wait_for_gate(end.turn_id, end.turn_revision):
            if end.response_key is None:
                return False
            end.cleanup_only = True
        return True

    def _admit_input(
        self,
        tts_input: TTSInput,
        *,
        commit: bool = False,
        hold_for_grace: bool = False,
    ) -> bool:
        """Wait for the speculative turn gate and say whether to synthesize ``tts_input``.

        Defaults to speculative admission (``commit=False, hold_for_grace=False``) so
        synthesis can overlap the reopen grace while ``BaseTTSHandler`` commits on the
        first yielded audio block.
        """
        tracker = self.speculative_turns
        if tracker is not None and not tracker.wait_for_gate(
            tts_input.turn_id,
            tts_input.turn_revision,
            hold_for_grace=hold_for_grace,
            commit=commit,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return False
        return True

    def _is_commit_cancelled(self, tts_input: TTSInput | None) -> bool:
        """Handler-specific cancellation check consulted while waiting to commit first audio."""
        return False

    def _assistant_text_for_print(self, tts_input: TTSInput) -> str:
        """Return the text printed to the terminal once first audio commits."""
        return tts_input.text

    def _on_first_audio_committed(
        self,
        tts_input: TTSInput,
        timer: TTSResponseTimer | None,
        e2e_s: float | None = None,
    ) -> None:
        """Run user-visible first-audio side effects after speculative commit succeeds."""
        if self._print_assistant_text:
            console.print(f"[green]ASSISTANT: {self._assistant_text_for_print(tts_input)}")

    def _commit_first_audio(
        self,
        tts_input: TTSInput | None,
        *,
        generation: int | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """Wait for the reopen grace / Smart Turn gate and commit before yielding first audio."""
        effective_gen = generation
        if effective_gen is None and tts_input is not None:
            effective_gen = tts_input.cancel_generation
        if effective_gen is None and self.cancel_scope is not None:
            effective_gen = self.cancel_scope.generation
        stop_event = getattr(self, "stop_event", None)

        def _aborted() -> bool:
            return (
                (stop_event is not None and stop_event.is_set())
                or self._is_stale(effective_gen, tts_input)
                or (
                    tts_input is not None
                    and tts_input.cancel_generation is not None
                    and self._is_stale(tts_input.cancel_generation, tts_input)
                )
                or self._is_commit_cancelled(tts_input)
                or (cancelled is not None and cancelled())
            )

        if _aborted():
            return False
        tracker = self.speculative_turns
        if tracker is not None and self._turn_uncommitted(tts_input):
            assert tts_input is not None
            # Suspend the MLX lock around any wait for an uncommitted turn so a
            # concurrent pending reopen cannot starve MLX STT behind this gate.
            with suspend_mlx_lock():
                committed = tracker.wait_for_gate(
                    tts_input.turn_id,
                    tts_input.turn_revision,
                    commit=True,
                    should_abort=_aborted,
                )
            if not committed:
                logger.debug(
                    "Dropping speculative TTS audio before commit for turn=%s rev=%s",
                    tts_input.turn_id,
                    tts_input.turn_revision,
                )
                return False
        return not _aborted()

    def _is_stale(self, generation: int | None, tts_input: TTSInput | None = None) -> bool:
        """True once an interruption or speculative turn reopen supersedes this work."""
        if generation is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(generation):
            return True
        active = tts_input if tts_input is not None else self._active_tts_input
        if active is not None and active.turn_id is not None and active.turn_revision is not None:
            is_latest = getattr(self.speculative_turns, "is_latest", None)
            if callable(is_latest) and not is_latest(active.turn_id, active.turn_revision):
                return True
        return False

    def _turn_uncommitted(self, tts_input: TTSInput | None) -> bool:
        tracker = self.speculative_turns
        if tracker is None or tts_input is None or tts_input.turn_id is None or tts_input.turn_revision is None:
            return False
        is_committed = getattr(tracker, "is_committed", None)
        return not (callable(is_committed) and is_committed(tts_input.turn_id, tts_input.turn_revision))

    def _latency_timer(
        self,
        tts_input: TTSInput | None,
        generation: int | None = None,
        *,
        also_cancelled: Callable[[], bool] | None = None,
    ) -> TTSResponseTimer:
        """Start timing one synthesis; call right before provider work begins.

        Output counts as cancelled once ``generation`` is stale, the speculative
        turn is superseded, or when ``also_cancelled`` returns True. For an
        uncommitted speculative turn, ``tts_ttfa`` and ``e2e`` stay deferred
        until ``_commit_first_audio`` succeeds so dropped speculative audio never
        tags a cancelled turn as ``tts=cut``.
        """

        def cancelled() -> bool:
            return self._is_stale(generation, tts_input) or (also_cancelled is not None and also_cancelled())

        timer = TTSResponseTimer(
            self.turn_latency_store,
            tts_input.response_key if tts_input is not None else None,
            tts_input.speech_stopped_at_s if tts_input is not None else None,
            cancelled,
            defer_until_commit=self._turn_uncommitted(tts_input),
        )
        if tts_input is not None and self._active_tts_input is tts_input:
            self._active_timer = timer
        return timer
