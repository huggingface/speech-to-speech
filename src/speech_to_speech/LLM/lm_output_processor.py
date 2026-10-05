"""
LLM Output Processor

Intercepts LLM output to:
1. Preserve text and tool events in model order
2. Forward clean text to TTS in the same queue
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from queue import Queue
from uuid import uuid4

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline import language_detection
from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantResponseDoneEvent,
    AssistantToolCallReadyEvent,
    PipelineEvent,
    ResponseFailedEvent,
    ResponseGenerationDoneEvent,
    TokenUsageEvent,
)
from speech_to_speech.pipeline.handler_types import LLMOut, TTSIn
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
    EndOfResponse,
    LLMResponseChunk,
    TokenUsage,
    TTSInput,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import transcript_for_log
from speech_to_speech.utils.utils import response_wants_audio

logger = logging.getLogger(__name__)


class LMOutputProcessor(BaseHandler[LLMOut, TTSIn | PipelineEvent]):
    """
    Places ordered output events and their TTS inputs on one queue.

    Input: :class:`LLMResponseChunk`, :class:`TokenUsage`, or :class:`EndOfResponse` from LLM
    Output: assistant and usage events, :class:`TTSInput`, or :class:`EndOfResponse`
    on the same ordered path through TTS.
    """

    def setup(
        self,
        speculative_turns: SpeculativeTurnTracker | None = None,
        text_output_queue: Queue[PipelineEvent] | None = None,
        detect_llm_output_language: bool = False,
    ) -> None:
        self.speculative_turns = speculative_turns
        self.text_output_queue = text_output_queue
        self.detect_llm_output_language = detect_llm_output_language
        self._language_detector = language_detection.warm_language_detector() if detect_llm_output_language else None
        self._response_key: str | None = None
        self._tool_call_ids: list[str] = []
        self._output_sequence = 0
        self._detected_assistant_language: str | None = None
        self._tts_runtime_config: RuntimeConfig | None = None
        self._response_selected_language: str | None = None
        self._response_tts_language: str | None = None
        self._response_language_resolved = False
        self._assistant_language_probe = ""

    def _start_response(self, response_key: str | None) -> str:
        key = response_key or self._response_key or uuid4().hex
        self._response_key = key
        return key

    def _reset_response(self) -> None:
        self._response_key = None
        self._tool_call_ids = []
        self._output_sequence = 0
        self._detected_assistant_language = None
        self._tts_runtime_config = None
        self._response_selected_language = None
        self._response_tts_language = None
        self._response_language_resolved = False
        self._assistant_language_probe = ""

    def _observe_assistant_language(self, text: str) -> None:
        if self._detected_assistant_language is not None:
            return
        # A few short streamed parts can make one detectable sentence. Keep only
        # a bounded window for this response.
        self._assistant_language_probe = (self._assistant_language_probe + " " + text).strip()[-256:]
        # The assistant can answer outside Parakeet's recognition languages.
        if self._language_detector is None:
            self._language_detector = language_detection.warm_language_detector()
        try:
            self._detected_assistant_language = language_detection.detect_language_from_text(
                self._assistant_language_probe,
                self._language_detector,
                minimum_confidence_gap=language_detection.MIN_ASSISTANT_CONFIDENCE_GAP,
                allow_short_cjk=True,
            )
        except Exception:
            logger.exception("Assistant language detection failed; using prior assistant language")

    def _notify_generation_done(
        self,
        lm_output: EndOfResponse,
        response_key: str | None,
        *,
        succeeded: bool,
    ) -> None:
        if self.text_output_queue is None:
            return
        self.text_output_queue.put(
            ResponseGenerationDoneEvent(
                response_key=response_key,
                call_ids=list(self._tool_call_ids),
                succeeded=succeeded,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
            )
        )

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def process(self, lm_output: LLMOut) -> Iterator[TTSIn | PipelineEvent]:
        """
        Forward response events and audio inputs in their original order.

        Yields:
            Response events, :class:`TTSInput`, or :class:`EndOfResponse`
        """
        if isinstance(lm_output, TokenUsage):
            usage_response_key = self._start_response(lm_output.response_key)
            usage_event = TokenUsageEvent(
                input_tokens=lm_output.input_tokens or 0,
                output_tokens=lm_output.output_tokens or 0,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=usage_response_key,
            )
            yield usage_event
            return

        if isinstance(lm_output, EndOfResponse):
            response_key = lm_output.response_key or self._response_key
            if not self._turn_output_allowed(
                lm_output.turn_id,
                lm_output.turn_revision,
            ):
                logger.debug(
                    "Dropping stale end-of-response for turn=%s rev=%s",
                    lm_output.turn_id,
                    lm_output.turn_revision,
                )
                self._notify_generation_done(lm_output, response_key, succeeded=False)
                self._reset_response()
                if response_key is not None:
                    # Bypass downstream speculative gates with a lifecycle-only
                    # terminal. The router uses its key to cancel an opened stale
                    # response or clear only that queued response.
                    yield EndOfResponse(
                        cancel_generation=lm_output.cancel_generation,
                        response_key=response_key,
                        cleanup_only=True,
                    )
                return
            succeeded = lm_output.error is None and lm_output.status == "completed"
            self._notify_generation_done(lm_output, response_key, succeeded=succeeded)
            if succeeded and self._detected_assistant_language and self._tts_runtime_config is not None:
                self._tts_runtime_config.last_assistant_language = self._detected_assistant_language
            if lm_output.error:
                yield ResponseFailedEvent(
                    message=lm_output.error,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
                    response_key=response_key,
                )
            else:
                yield AssistantResponseDoneEvent(
                    status=lm_output.status,
                    reason=lm_output.reason,
                    response_key=response_key,
                    turn_id=lm_output.turn_id,
                    turn_revision=lm_output.turn_revision,
                    cancel_generation=lm_output.cancel_generation,
                )
            yield EndOfResponse(
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
            )
            self._reset_response()
            return

        if not isinstance(lm_output, LLMResponseChunk):
            logger.warning("LMOutputProcessor received unexpected type: %s", type(lm_output))
            return

        if not self._turn_output_allowed(
            lm_output.turn_id,
            lm_output.turn_revision,
        ):
            logger.debug("Dropping stale LLM chunk for turn=%s rev=%s", lm_output.turn_id, lm_output.turn_revision)
            return

        logger.debug("LM processor: parts=%s", transcript_for_log(lm_output.parts))

        response_key = self._start_response(lm_output.response_key)

        for part in lm_output.parts:
            output_sequence = self._output_sequence
            self._output_sequence += 1
            if isinstance(part, AssistantToolCallPart) and part.tool.call_id not in self._tool_call_ids:
                self._tool_call_ids.append(part.tool.call_id)
                if self.text_output_queue is not None:
                    self.text_output_queue.put(
                        AssistantToolCallReadyEvent(
                            part=part,
                            output_sequence=output_sequence,
                            turn_id=lm_output.turn_id,
                            turn_revision=lm_output.turn_revision,
                            cancel_generation=lm_output.cancel_generation,
                            response_key=response_key,
                        )
                    )
            event = AssistantOutputEvent(
                parts=[part],
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
                output_sequence=output_sequence,
            )
            yield event
            if (
                not isinstance(part, AssistantTextPart)
                or not part.text.strip()
                or not response_wants_audio(lm_output.response)
            ):
                continue
            logger.debug("Forwarding to TTS: %s", transcript_for_log(part.text))
            config = lm_output.runtime_config
            if self.detect_llm_output_language or config is not None:
                self._observe_assistant_language(part.text)
            self._tts_runtime_config = config
            language_code = lm_output.language_code
            if self.detect_llm_output_language:
                language_code = self._detected_assistant_language
            if not self._response_language_resolved:
                self._response_selected_language = (
                    lm_output.selected_language
                    if "selected_language" in lm_output.model_fields_set
                    else config.selected_language
                    if config is not None
                    else None
                )
                selected = self._response_selected_language
                if selected == "auto" or (selected is None and self.detect_llm_output_language):
                    self._response_tts_language = self._detected_assistant_language or (
                        config.last_assistant_language if config is not None else None
                    )
                elif selected is not None:
                    # STT keeps the named selection. TTS follows a confident
                    # detection of this response's first spoken text, and
                    # otherwise the selection. TTS keeps the selection for a
                    # detected language it cannot speak.
                    detected = self._detected_assistant_language if self.detect_llm_output_language else None
                    self._response_tts_language = detected or selected
                self._response_language_resolved = True
            selected = self._response_selected_language
            yield TTSInput(
                text=part.text,
                language_code=language_code,
                selected_language=selected,
                tts_language_code=self._response_tts_language if selected is not None else language_code,
                response_assistant_language_code=(
                    self._response_tts_language if selected is None and self.detect_llm_output_language else None
                ),
                runtime_config=lm_output.runtime_config,
                response=lm_output.response,
                turn_id=lm_output.turn_id,
                turn_revision=lm_output.turn_revision,
                speech_stopped_at_s=lm_output.speech_stopped_at_s,
                cancel_generation=lm_output.cancel_generation,
                response_key=response_key,
                prefetch_transaction=lm_output.prefetch_transaction,
            )

    def on_session_end(self) -> None:
        self._reset_response()
