from __future__ import annotations

import base64
import io
import logging
import time
import wave
from collections.abc import Iterator
from typing import Any, Optional, cast

import httpx
import numpy as np
from nltk import sent_tokenize
from openai import Stream
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseTextDeltaEvent,
)

from speech_to_speech.LLM.base_openai_compatible_language_model import (
    WARMUP_MAX_RETRIES,
    AssistantMessage,
    BaseOpenAICompatibleHandler,
    ProviderEvent,
    TextDelta,
    ToolCall,
    Usage,
)
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    build_active_chat,
    make_assistant_message,
    make_user_audio_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import EndOfResponse, LLMResponseChunk, TokenUsage
from speech_to_speech.utils.utils import _generate_id, is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)


class ResponsesApiModelHandler(BaseOpenAICompatibleHandler):
    """LLM handler that talks to an OpenAI ``/v1/responses`` server."""

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        start = time.time()
        self.client.with_options(max_retries=WARMUP_MAX_RETRIES).responses.create(
            model=self.model_name,
            input=[
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are a helpful assistant"}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            timeout=self.request_timeout,
        )
        end = time.time()
        logger.info(f"{self.__class__.__name__}:  warmed up! time: {(end - start):.3f} s")

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a generate fn that calls the Responses API for compaction."""
        client = self.client
        model_name = self.model_name
        timeout = self.request_timeout

        def generate(system: str, user: str) -> str:
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    },
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user}],
                    },
                ],
                timeout=timeout,
            )
            return response.output_text

        return generate

    @staticmethod
    def _audio_to_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
        audio_array = np.asarray(audio)
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        if np.issubdtype(audio_array.dtype, np.floating):
            pcm = (np.clip(audio_array, -1.0, 1.0) * 32767.0).astype("<i2")
        else:
            pcm = np.clip(audio_array, -32768, 32767).astype("<i2")

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())
            return base64.b64encode(wav_io.getvalue()).decode("ascii")

    @staticmethod
    def _chat_usage_tokens(usage: Any) -> tuple[int, int]:
        if usage is None:
            return 0, 0
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", 0)
        return int(input_tokens or 0), int(output_tokens or 0)

    def _audio_chat_kwargs(self, response: Any, optional_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(optional_kwargs)
        max_tokens = getattr(response, "max_output_tokens", None) if response is not None else None
        kwargs.setdefault("max_tokens", max_tokens or self.audio_max_tokens)
        kwargs.setdefault("temperature", self.audio_temperature)
        return kwargs

    def _generate_audio_chat_completions(
        self,
        active_chat: Chat,
        original_chat: Chat,
        request_audio: np.ndarray,
        audio_sample_rate: int,
        language_code: Optional[str],
        gen: int | None,
        runtime_config: Any,
        response: Any,
        optional_kwargs: dict[str, Any],
        turn_id: str | None,
        turn_revision: int | None,
        speech_stopped_at_s: float | None,
    ) -> Iterator[LLMOut]:
        audio_message = make_user_audio_message(self._audio_to_wav_base64(request_audio, audio_sample_rate))
        active_chat.add_item(audio_message)
        consumed_image_ids = active_chat.image_message_ids()

        api_response: Any = None
        clean_text = ""
        input_tokens = 0
        output_tokens = 0
        cancelled = False
        error_message: str | None = None
        try:
            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=cast(Any, active_chat.to_chat_completions_chat()),
                stream=self.stream,
                extra_body=self._extra_body,
                timeout=self.request_timeout,
                **self._audio_chat_kwargs(response, optional_kwargs),
            )
            if self.stream:
                printable_text = ""
                sentence_batch: list[str] = []
                for raw_event in api_response:
                    if self._generation_is_stale(gen) or not self._turn_is_latest(turn_id, turn_revision):
                        logger.info("Audio LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    usage = getattr(raw_event, "usage", None)
                    if usage:
                        input_tokens, output_tokens = self._chat_usage_tokens(usage)
                    if not getattr(raw_event, "choices", None):
                        continue
                    delta = raw_event.choices[0].delta
                    new_text = remove_unspeechable(getattr(delta, "content", None) or "")
                    if not new_text:
                        continue
                    clean_text += new_text
                    printable_text += new_text
                    sentences = sent_tokenize(printable_text)
                    if len(sentences) > 1:
                        for sentence in sentences[:-1]:
                            sentence_batch.append(sentence)
                            if len(sentence_batch) >= self.stream_batch_sentences:
                                if not self._turn_output_allowed(turn_id, turn_revision):
                                    logger.info("Audio LLM generation cancelled (stale speculative turn)")
                                    cancelled = True
                                    break
                                yield LLMResponseChunk(
                                    text=" ".join(sentence_batch),
                                    language_code=language_code,
                                    runtime_config=runtime_config,
                                    response=response,
                                    turn_id=turn_id,
                                    turn_revision=turn_revision,
                                    speech_stopped_at_s=speech_stopped_at_s,
                                    cancel_generation=gen,
                                )
                                sentence_batch = []
                        if cancelled:
                            break
                        printable_text = sentences[-1]
                if not cancelled:
                    if printable_text.strip():
                        sentence_batch.append(printable_text.strip())
                    remaining = " ".join(sentence_batch)
                    if (
                        remaining
                        and not self._generation_is_stale(gen)
                        and self._turn_output_allowed(turn_id, turn_revision)
                    ):
                        yield LLMResponseChunk(
                            text=remaining,
                            language_code=language_code,
                            runtime_config=runtime_config,
                            response=response,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                            speech_stopped_at_s=speech_stopped_at_s,
                            cancel_generation=gen,
                        )
            else:
                usage = getattr(api_response, "usage", None)
                input_tokens, output_tokens = self._chat_usage_tokens(usage)
                if self._generation_is_stale(gen) or not self._turn_is_latest(turn_id, turn_revision):
                    logger.info("Audio LLM generation cancelled (interruption)")
                    cancelled = True
                elif getattr(api_response, "choices", None):
                    content = api_response.choices[0].message.content or ""
                    clean_text = remove_unspeechable(content).strip()
                    if clean_text and self._turn_output_allowed(turn_id, turn_revision):
                        yield LLMResponseChunk(
                            text=clean_text,
                            language_code=language_code,
                            runtime_config=runtime_config,
                            response=response,
                            turn_id=turn_id,
                            turn_revision=turn_revision,
                            speech_stopped_at_s=speech_stopped_at_s,
                            cancel_generation=gen,
                        )
        except httpx.ReadTimeout:
            logger.warning(
                "OpenAI-compatible audio chat read timed out after %.1fs; ending the current response",
                self.request_timeout_s,
            )
            cancelled = True
            if not self._generation_is_stale(gen) and self._turn_output_allowed(turn_id, turn_revision):
                yield LLMResponseChunk(
                    text="Wow I'm a bit slow today, could you repeat that?",
                    runtime_config=runtime_config,
                    response=response,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    speech_stopped_at_s=speech_stopped_at_s,
                    cancel_generation=gen,
                )
        except Exception as exc:
            logger.exception("Audio LLM generation failed; ending the current response")
            error_message = f"Language model generation failed: {exc}"
        finally:
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

        if (
            error_message is None
            and not cancelled
            and not self._generation_is_stale(gen)
            and self._turn_output_allowed(turn_id, turn_revision)
        ):
            if not is_out_of_band(response):
                original_chat.add_item(audio_message)
                if clean_text.strip():
                    original_chat.add_item(make_assistant_message(clean_text.strip()))
                original_chat.strip_audio()
                original_chat.strip_images(consumed_image_ids)
                original_chat.trim_if_needed(self.compactor)
            if input_tokens or output_tokens:
                yield TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                )
        yield EndOfResponse(
            turn_id=turn_id,
            turn_revision=turn_revision,
            cancel_generation=gen,
            error=error_message,
        )

    # ── base hooks ──────────────────────────────────────────────────────────--

    def _serialize(self, active_chat: Chat) -> Any:
        return active_chat.to_responses_api_chat()

    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        optional_kwargs: dict[str, Any] = {}
        if req_tools is not None:
            optional_kwargs["tools"] = req_tools
        if req_tool_choice is not None:
            optional_kwargs["tool_choice"] = req_tool_choice
        return optional_kwargs

    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        return self.client.responses.create(
            model=self.model_name,
            input=api_input,
            stream=self.stream,
            extra_body=self._extra_body,
            timeout=self.request_timeout,
            **optional_kwargs,
        )

    @staticmethod
    def _assistant_content(content: Any) -> list[AssistantContent]:
        return [
            AssistantContent(type="output_text", text=c.text if c.type == "output_text" else c.refusal) for c in content
        ]

    def _iter_stream_events(self, api_response: Stream) -> Iterator[ProviderEvent]:
        for raw_event in api_response:
            if isinstance(raw_event, ResponseTextDeltaEvent):
                yield TextDelta(text=raw_event.delta)
            elif isinstance(raw_event, ResponseOutputItemDoneEvent):
                item = raw_event.item
                if isinstance(item, ResponseFunctionToolCall):
                    item.call_id = _generate_id("call")
                    item.id = _generate_id("fc")
                    yield ToolCall(item=item)
                elif isinstance(item, ResponseOutputMessage):
                    yield AssistantMessage(content=self._assistant_content(item.content))
            elif isinstance(raw_event, ResponseCompletedEvent):
                usage = getattr(raw_event.response, "usage", None)
                if usage:
                    yield Usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0)

    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        usage = api_response.usage
        if usage:
            yield Usage(input_tokens=usage.input_tokens or 0, output_tokens=usage.output_tokens or 0)
        for message in api_response.output:
            if isinstance(message, ResponseFunctionToolCall):
                message.call_id = _generate_id("call")
                message.id = _generate_id("fc")
                yield ToolCall(item=message)
            elif isinstance(message, ResponseOutputMessage):
                yield AssistantMessage(content=self._assistant_content(message.content))
                # Text-only keeps every character; the base applies remove_unspeechable
                # for audio. Only output_text parts are spoken (refusals are stored).
                raw = "".join(c.text for c in message.content if c.type == "output_text")
                yield TextDelta(text=raw)
            else:
                logger.warning(f"Not supported message type: {message.type}")

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        if request.audio is None:
            yield from super().process(request)
            return

        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision)
            return

        original_chat = runtime_config.chat
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=turn_id, turn_revision=turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()

        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        ) or ""
        req_tools = response.tools if response and response.tools else runtime_config.session.tools
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        self._apply_config(active_chat, instructions, response_wants_audio(response))
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)
        gen = self.cancel_scope.generation if self.cancel_scope else None
        yield from self._generate_audio_chat_completions(
            active_chat,
            original_chat,
            request.audio,
            request.audio_sample_rate,
            language_code,
            gen,
            runtime_config,
            response,
            optional_kwargs,
            turn_id,
            turn_revision,
            speech_stopped_at_s,
        )

    def on_session_end(self) -> None:
        logger.debug("OpenAI API language model session state reset")
