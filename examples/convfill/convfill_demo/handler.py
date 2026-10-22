from __future__ import annotations

import logging
import os
import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Iterator

from strip_markdown import strip_markdown

from convfill_demo.runtime.run_convfill import ConvFillConfig, ConvFillSystem
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import make_assistant_message
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk
from speech_to_speech.utils.utils import is_out_of_band

from .delivery import DeliveryLedgerQueue

logger = logging.getLogger(__name__)

_TurnEvent = tuple[str, Any]
_SystemFactory = Callable[..., Any]


def latest_user_text(chat: Any) -> str:
    """Return the newest text user message from an HF realtime Chat snapshot."""
    for message in reversed(chat.to_transformers_chat()):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(part.get("text", "")).strip()
                for part in content
                if isinstance(part, dict) and part.get("type") == "input_text"
            ]
            return " ".join(part for part in parts if part).strip()
    return ""


class ConvFillRealtimeHandler(BaseHandler[LLMIn, LLMOut]):
    """Use the existing Talker/Reasoner system as an HF realtime LLM stage.

    The handler consumes the final STT transcript already stored in the
    realtime session chat. ConvFill runs the remote Reasoner and fine-tuned Talker concurrently; only Talker fragments leave this stage.
    """

    def setup(
        self,
        config_path: str = "configs/demo_mode/convfill_hosted_config.json",
        mode: str = "normal",
        cancel_scope: Any = None,
        speculative_turns: Any = None,
        system_factory: _SystemFactory | None = None,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self._active_events: Queue[_TurnEvent] | None = None
        self._audio_lock = threading.Lock()
        self._pending_audio_fragments = 0
        self._yielded_fragments = 0
        self._response_key: str | None = None
        self._ledger: DeliveryLedgerQueue | None = None
        self._turn_started_at = None
        self._first_knowledge = False

        callbacks = {
            "on_thought": self._on_thought,
            "on_response_fragment": self._on_response_fragment,
            "on_turn_complete": self._on_turn_complete,
            "on_frontend_inference": self._on_frontend_inference,
            "on_phrase_start": self._on_phrase_start,
            "emit_tts": False,
            "audio_in_flight_fn": self._audio_in_flight,
        }
        if system_factory is not None:
            self.system = system_factory(**callbacks)
        else:
            config = ConvFillConfig(config_path, mode=mode)
            self.system = ConvFillSystem(config, **callbacks)

    def set_delivery_ledger(self, ledger: DeliveryLedgerQueue) -> None:
        """Connect the pipeline delivery ledger to ConvFill's silence scheduler."""
        self._ledger = ledger
        self.system.set_audio_in_flight_fn(self._audio_in_flight)

    def _audio_in_flight(self) -> int:
        ledger = self._ledger
        if ledger is None:
            # No transport attached (unit tests, or a caller-supplied system):
            # fall back to counting fragments handed downstream.
            with self._audio_lock:
                return self._pending_audio_fragments
        with self._audio_lock:
            yielded = self._yielded_fragments
            response_key = self._response_key
        if yielded > len(ledger.delivered_text(response_key)):
            # Sentences are in the pipeline but have not reached the client yet.
            return 1
        if ledger.pending_audio() or ledger.audio_playing():
            return 1
        return 0

    def _delivered_text(self, response_key: str | None) -> str:
        """Everything the client was sent for *response_key*, joined."""
        if self._ledger is None:
            return ""
        parts = [text.strip() for text in self._ledger.delivered_text(response_key)]
        return " ".join(part for part in parts if part).strip()

    def _release_ledger(self, response_key: str | None) -> None:
        if self._ledger is None:
            return
        self._ledger.reset_playback()
        self._ledger.clear_response(response_key)

    def _put(self, kind: str, payload: Any = None) -> None:
        if self._active_events is not None:
            self._active_events.put((kind, payload))

    def _on_thought(self, thought: str) -> None:
        if thought != "<sil>" and self._turn_started_at is not None and not self._first_knowledge:
            self._first_knowledge = True
            logger.info("ConvFill first useful knowledge: %.3fs", time.perf_counter() - self._turn_started_at)
        if thought != "<sil>" and os.environ.get("CONVFILL_LOG_CONTENT") == "1":
            logger.info("ConvFill Reasoner knowledge: %s", thought)

    def _on_response_fragment(self, text: str) -> None:
        if getattr(self.system, "cancel_event", threading.Event()).is_set():
            return
        cleaned = strip_markdown(text).strip()
        if not cleaned:
            return
        with self._audio_lock:
            self._pending_audio_fragments += 1
        self._put("fragment", cleaned)

    def _on_turn_complete(self, final_response: str, final_thoughts: list[str]) -> None:
        # ConvFill has already committed this turn to its dialogue manager.
        # ``process`` waits for ``run_turn`` itself, so no extra queue sentinel
        # is needed here.
        return

    def _on_frontend_inference(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def _on_phrase_start(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def _turn_is_stale(self, request: Any, generation: int | None) -> bool:
        if generation is not None and self.cancel_scope is not None:
            if self.cancel_scope.is_stale(generation):
                return True
        if self.speculative_turns is not None:
            return not self.speculative_turns.is_latest(request.turn_id, request.turn_revision)
        return False

    @staticmethod
    def _chunk(request: Any, generation: int | None, text: str) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=request.language_code,
            runtime_config=request.runtime_config,
            response=request.response,
            turn_id=request.turn_id,
            turn_revision=request.turn_revision,
            speech_stopped_at_s=request.speech_stopped_at_s,
            cancel_generation=generation,
            response_key=request.response_key,
            prefetch_transaction=request.prefetch_transaction,
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        generation = self.cancel_scope.generation if self.cancel_scope is not None else None
        if is_out_of_band(request.response):
            yield EndOfResponse(
                turn_id=request.turn_id,
                turn_revision=request.turn_revision,
                cancel_generation=generation,
                response_key=request.response_key,
                error="ConvFill does not support out-of-band responses.",
            )
            return

        user_text = latest_user_text(request.runtime_config.chat)
        if not user_text:
            yield EndOfResponse(
                turn_id=request.turn_id,
                turn_revision=request.turn_revision,
                cancel_generation=generation,
                response_key=request.response_key,
                error="ConvFill received no transcribed user text.",
            )
            return

        events: Queue[_TurnEvent] = Queue()
        self._active_events = events
        with self._audio_lock:
            self._pending_audio_fragments = 0
            self._yielded_fragments = 0
            self._response_key = request.response_key

        def run_turn() -> None:
            try:
                completed = bool(self.system.run_turn(user_text))
                events.put(("runner_done", completed))
            except BaseException as exc:
                events.put(("runner_error", exc))

        self._turn_started_at = time.perf_counter()
        self._first_knowledge = False
        worker = threading.Thread(target=run_turn, daemon=True, name="convfill-realtime-turn")
        worker.start()

        emitted: list[str] = []
        completed = False
        cancelled = False
        error: BaseException | None = None
        deadline = (
            time.monotonic() + float(getattr(getattr(self.system, "convfill_config", None), "turn_timeout_s", 90)) + 5
        )
        try:
            while worker.is_alive() or not events.empty():
                if time.monotonic() > deadline:
                    self.system.cancel_turn()
                    error = TimeoutError("ConvFill handler deadline exceeded")
                    break
                if self._turn_is_stale(request, generation):
                    if not cancelled:
                        cancelled = True
                        self.system.cancel_turn()
                        # The client drops its playback buffer on barge-in, so
                        # anything still scheduled is never heard.
                        if self._ledger is not None:
                            self._ledger.reset_playback()
                    elif not getattr(self.system, "cancel_event", threading.Event()).is_set():
                        # Cancellation can race run_turn's new event creation.
                        self.system.cancel_turn()
                try:
                    kind, payload = events.get(timeout=0.05)
                except Empty:
                    continue

                if kind == "fragment":
                    if not cancelled:
                        emitted.append(str(payload))
                        with self._audio_lock:
                            self._yielded_fragments += 1
                        yield self._chunk(request, generation, str(payload))
                elif kind == "runner_done":
                    completed = bool(payload)
                elif kind == "runner_error":
                    error = payload

            worker.join(timeout=2.0)

            if self._turn_is_stale(request, generation):
                cancelled = True

            heard = ""
            if cancelled and error is None:
                # Best-effort dispatched-text prefix. The transport has no
                # playback acknowledgement: this is NOT verified heard text.
                # The interruption marker tells the Reasoner it may be partial.
                heard = self._delivered_text(request.response_key)
                if heard:
                    self.system.commit_interrupted_turn(user_text, heard)
                else:
                    self.system.discard_pending_turn(user_text)
                completed = False

            if completed and emitted and not is_out_of_band(request.response):
                assistant = make_assistant_message(" ".join(emitted).strip())
                recorded = request.runtime_config.chat.add_provisional_generation_items(
                    request.response_key,
                    [assistant],
                )
                if recorded is None:
                    self.system.dialogue_state_manager.rollback_turn(user_text)
                    completed = False
                else:
                    request.runtime_config.chat.trim_if_needed()
            elif heard and not is_out_of_band(request.response):
                # Best effort: HF may refuse a provisional item for a cancelled
                # response. ConvFill keeps its own commit either way.
                try:
                    request.runtime_config.chat.add_provisional_generation_items(
                        request.response_key,
                        [make_assistant_message(heard)],
                    )
                except Exception:
                    logger.debug("HF chat refused the interrupted turn text", exc_info=True)

            error_message = None
            if error is not None:
                self.system.dialogue_state_manager.rollback_turn(user_text)
                completed = False
                logger.error(
                    "ConvFill realtime turn failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                error_message = "ConvFill inference failed. Please try again."
            elif not cancelled and not completed:
                self.system.dialogue_state_manager.rollback_turn(user_text)
                error_message = "ConvFill did not complete the response."
            elif completed and not emitted:
                self.system.dialogue_state_manager.rollback_turn(user_text)
                completed = False
                error_message = "ConvFill completed without producing a Talker response."

            yield EndOfResponse(
                turn_id=request.turn_id,
                turn_revision=request.turn_revision,
                cancel_generation=generation,
                response_key=request.response_key,
                error=error_message,
            )
        finally:
            if worker.is_alive():
                self.system.cancel_turn()
                worker.join(timeout=2.0)
            self._active_events = None
            self._turn_started_at = None
            self._release_ledger(request.response_key)
            with self._audio_lock:
                self._pending_audio_fragments = 0
                self._yielded_fragments = 0
                self._response_key = None

    def on_session_end(self) -> None:
        self.system.reset()
        if self._ledger is not None:
            self._ledger.reset_playback()
        with self._audio_lock:
            self._pending_audio_fragments = 0
            self._yielded_fragments = 0
            self._response_key = None
