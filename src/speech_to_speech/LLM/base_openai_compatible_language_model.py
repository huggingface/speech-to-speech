from __future__ import annotations

import ipaddress
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Lock, Thread, current_thread
from threading import Event as ThreadingEvent
from time import perf_counter
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import httpx
from nltk import sent_tokenize
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
    RealtimeConversationItemUserMessage,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseReasoningItem
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    ResponsesAssistantMessage,
    ResponsesFunctionCall,
    SupportedItem,
    build_active_chat,
    make_user_audio_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import (
    language_name_for_prompt,
    remove_markdown,
    remove_unspeechable,
    resolve_auto_language,
    sent_tokenize_preserving_markdown_code,
)
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.history import ResponseHistory
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    ResponseIncompleteReason,
    ResponsePrefetchTransaction,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.pipeline.transcript_logging import log_exception, transcript_for_log
from speech_to_speech.utils.utils import audio_to_wav_base64, is_out_of_band, response_wants_audio

logger = logging.getLogger(__name__)

# About 18–24 seconds of default SDK backoff before warmup fails.
WARMUP_MAX_RETRIES = 6
PREFETCH_PROVIDER_WORKER_LIMIT = 1
PREFETCH_STREAM_QUEUE_MAXSIZE = 16
PREFETCH_WORKER_ACQUIRE_TIMEOUT_S = 0.05
PROVIDER_FAILURE_FALLBACK = "I'm having trouble responding right now. Please try again."


# ── Normalised provider events ────────────────────────────────────────────────
# Each backend's stream/response is mapped to this small vocabulary so the shared
# speech-pipeline logic (sentence batching, cancellation, history, token usage)
# lives in one place. Subclasses differ only in how they produce these events.


class TextDelta(BaseModel):
    """Incremental assistant text. Always RAW (unfiltered); the base applies
    ``remove_unspeechable`` for the audio path."""

    text: str


class AssistantMessage(BaseModel):
    """A complete assistant turn to write back to history."""

    content: list[AssistantContent]
    id: str | None = None
    response_item: ResponseOutputMessage | None = None

    def to_chat_item(self) -> RealtimeConversationItemAssistantMessage:
        item = RealtimeConversationItemAssistantMessage(
            type="message", role="assistant", content=self.content, id=self.id
        )
        if self.response_item is not None:
            return ResponsesAssistantMessage(
                **item.model_dump(exclude_unset=True), response_item=self.response_item.model_copy(deep=True)
            )
        return item


class ToolCall(BaseModel):
    """A complete function tool call."""

    item: ResponseFunctionToolCall
    response_item: ResponseFunctionToolCall | None = None


class Usage(BaseModel):
    """Token accounting for the turn."""

    input_tokens: int
    output_tokens: int


class ProviderResponseEnd(BaseModel):
    """An explicit provider terminal, separate from stream exhaustion."""

    status: Literal["completed", "incomplete", "failed"] = "completed"
    reason: ResponseIncompleteReason | None = None
    error: str | None = None
    # Responses output_index order, independent of item completion timing.
    history_item_order: list[str] | None = None


ProviderEvent = TextDelta | AssistantMessage | ToolCall | ResponseReasoningItem | Usage | ProviderResponseEnd
SerializeFn = Callable[[Chat], Any]
RequestFn = Callable[[Any, dict[str, Any]], Any]
EventIteratorFn = Callable[[Any], Iterator[ProviderEvent]]


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    selected_language: str | None
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    speech_stopped_at_s: float | None
    wants_audio: bool
    response_key: str
    prefetch_transaction: ResponsePrefetchTransaction | None = None
    # End of the conversation when this turn started; keeps its output ahead of
    # user messages appended while the model was still running.
    history_anchor_id: str | None = None


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    history_items: list[SupportedItem] = Field(default_factory=list)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0
    output_emitted: bool = False
    ending: ProviderResponseEnd = Field(default_factory=ProviderResponseEnd)


class BaseOpenAICompatibleHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Shared lifecycle for OpenAI-compatible LLM backends (Responses & Chat
    Completions).

    Subclasses implement four hooks — :meth:`warmup`,
    :meth:`_build_compaction_generate_fn`, :meth:`_serialize`, :meth:`_request`,
    :meth:`_iter_events` and :meth:`_build_optional_kwargs` — and inherit the
    request/response orchestration: speculative-turn gating, cancellation,
    sentence batching, text-only vs audio handling, history write-back, token
    usage, out-of-band handling and error termination.
    """

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model_name: str = "gpt-5.6-terra",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        stream: bool = True,
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        disable_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        request_timeout_s: float = 20.0,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        audio_max_tokens: int = 256,
        audio_temperature: float = 0.0,
        audio_content_type: Literal["input_audio", "audio_url"] = "input_audio",
        audio_history_turns: int = 1,
        **_kwargs: Any,
    ) -> None:
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.model_name = model_name
        self.stream = stream
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt
        self.gen_kwargs = dict(gen_kwargs)
        self.audio_max_tokens = audio_max_tokens
        self.audio_temperature = audio_temperature
        if audio_content_type not in {"input_audio", "audio_url"}:
            raise ValueError("audio_content_type must be either 'input_audio' or 'audio_url'.")
        self.audio_content_type = audio_content_type
        self.audio_history_turns = max(0, audio_history_turns)
        self.reasoning_effort = reasoning_effort
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        self.user_role = user_role
        if (
            api_key is None
            and not os.environ.get("OPENAI_API_KEY")
            and base_url is not None
            and self._is_local_base_url(base_url)
        ):
            api_key = "none"
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self._prefetch_worker_slots = BoundedSemaphore(PREFETCH_PROVIDER_WORKER_LIMIT)
        self._prefetch_workers_lock = Lock()
        self._prefetch_workers: set[Thread] = set()
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether ``base_url`` points at the official OpenAI server.

        Normalises a trailing slash so ``https://api.openai.com/v1/`` is also
        recognised; the official server rejects the provider-specific extra_body
        keys we send to vLLM / the HF router.
        """
        if base_url is None:
            return False
        return base_url.rstrip("/") == "https://api.openai.com/v1"

    @staticmethod
    def _is_local_base_url(base_url: str) -> bool:
        """Whether *base_url* points at localhost or a loopback IP address."""
        host = urlparse(base_url).hostname
        if host is None:
            return False
        if host.rstrip(".").lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @classmethod
    def _build_extra_body(
        cls,
        base_url: Optional[str],
        disable_thinking: bool,
        reasoning_effort: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Build the provider-specific ``extra_body`` used to disable reasoning.

        Providers differ in how reasoning is turned off: vLLM/Qwen honour
        ``chat_template_kwargs.enable_thinking=false``, while others (e.g. GLM via
        the HF router) ignore that and require ``reasoning_effort='none'``. A
        non-empty ``reasoning_effort`` therefore takes precedence, including for
        official OpenAI requests; otherwise we fall back to the provider-specific
        chat-template flag, which the official OpenAI server does not accept.
        """
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        if base_url is None or cls._is_official_openai(base_url):
            return None
        if disable_thinking:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return None

    # ── subclass hooks ──────────────────────────────────────────────────────--

    @abstractmethod
    def warmup(self) -> None:
        """Issue a cheap request so the model/connection is ready before serving."""
        ...

    @abstractmethod
    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` fn used to compact long histories."""
        ...

    @abstractmethod
    def _serialize(self, active_chat: Chat) -> Any:
        """Serialise the chat to the backend's request payload (input/messages)."""
        ...

    @abstractmethod
    def _request(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        """Issue the create() call and return the response or stream."""
        ...

    @abstractmethod
    def _iter_stream_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a streaming response to normalised :data:`ProviderEvent`s."""
        ...

    @abstractmethod
    def _iter_response_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Map a non-streaming response to normalised :data:`ProviderEvent`s."""
        ...

    def _iter_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        """Dispatch to the stream/non-stream mapper. ``self.stream`` is the single
        source of truth (it set the request's ``stream=`` flag), so the response
        type always matches it."""
        if self.stream:
            yield from self._iter_stream_events(api_response)
        else:
            yield from self._iter_response_events(api_response)

    @abstractmethod
    def _build_optional_kwargs(self, req_tools: Any, req_tool_choice: Any) -> dict[str, Any]:
        """Build the per-request tools/tool_choice kwargs in the backend's shape."""
        ...

    # ── audio-input protocol hooks ───────────────────────────────────────────

    def _serialize_audio(self, active_chat: Chat) -> Any:
        """Serialize an audio turn using the selected backend's native protocol."""
        return self._serialize(active_chat)

    def _build_audio_optional_kwargs(
        self,
        response: Any,
        req_tools: Any,
        req_tool_choice: Any,
    ) -> dict[str, Any]:
        """Build audio request parameters in the selected backend's shape."""
        kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)
        max_tokens = getattr(response, "max_output_tokens", None) if response is not None else None
        kwargs.setdefault("max_tokens", max_tokens or self.audio_max_tokens)
        kwargs.setdefault("temperature", self.audio_temperature)
        return kwargs

    def _request_audio(self, api_input: Any, optional_kwargs: dict[str, Any]) -> Any:
        return self._request(api_input, optional_kwargs)

    def _iter_audio_events(self, api_response: Any) -> Iterator[ProviderEvent]:
        yield from self._iter_events(api_response)

    _audio_to_wav_base64 = staticmethod(audio_to_wav_base64)

    def cleanup(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            del self.client
            client.close()

    # ── speculative-turn / cancellation gating ─────────────────────────────────

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _generation_is_stale(self, gen: int | None) -> bool:
        return gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen)

    def _turn_is_cancelled(self, turn: _Turn) -> bool:
        return (
            turn.prefetch_transaction is not None
            and turn.prefetch_transaction.discarded
            or self._generation_is_stale(turn.gen)
        )

    @staticmethod
    def _close_response(response: Any) -> None:
        if response is not None and hasattr(response, "close"):
            try:
                response.close()
            except Exception:
                pass

    def _start_prefetch_worker(self, target: Callable[[], None], *, name: str) -> Thread | None:
        """Start one tracked provider worker without exceeding the fixed cap."""
        if not self._prefetch_worker_slots.acquire(timeout=PREFETCH_WORKER_ACQUIRE_TIMEOUT_S):
            return None

        def run() -> None:
            try:
                target()
            finally:
                worker = current_thread()
                with self._prefetch_workers_lock:
                    self._prefetch_workers.discard(worker)
                self._prefetch_worker_slots.release()

        worker = Thread(target=run, name=name, daemon=True)
        with self._prefetch_workers_lock:
            self._prefetch_workers.add(worker)
        try:
            worker.start()
        except BaseException:
            with self._prefetch_workers_lock:
                self._prefetch_workers.discard(worker)
            self._prefetch_worker_slots.release()
            raise
        return worker

    def _iter_prefetch_events_interruptibly(
        self,
        request: Callable[[], Any],
        event_iterator: Callable[[Any], Iterator[ProviderEvent]],
        turn: _Turn,
    ) -> Iterator[ProviderEvent]:
        """Connect and consume one prefetch in a single bounded worker."""
        transaction = turn.prefetch_transaction
        assert transaction is not None
        results: Queue[tuple[bool, Any]] = Queue(maxsize=PREFETCH_STREAM_QUEUE_MAXSIZE)
        done = object()
        stop_reader = ThreadingEvent()
        response_lock = Lock()
        connected_response: list[Any] = []

        def reader_cancelled() -> bool:
            return stop_reader.is_set() or self._turn_is_cancelled(turn)

        def publish(result: tuple[bool, Any]) -> bool:
            while not reader_cancelled():
                try:
                    results.put(result, timeout=0.05)
                except Full:
                    continue
                return True
            return False

        def connect_and_read_events() -> None:
            api_response: Any = None
            try:
                api_response = request()
                with response_lock:
                    connected_response.append(api_response)
                if hasattr(api_response, "close"):
                    transaction.register_abort(api_response.close)
                if reader_cancelled():
                    return
                for event in event_iterator(api_response):
                    if not publish((True, event)):
                        return
            except BaseException as exc:
                publish((False, exc))
                return
            finally:
                self._close_response(api_response)
            publish((True, done))

        worker = self._start_prefetch_worker(connect_and_read_events, name="realtime-tool-prefetch")
        if worker is None:
            # discard() and claim() share one transaction lock. Calling discard
            # first makes the decision atomic: an already-claimed response stays
            # claimed, while a still-hidden one becomes permanently unclaimable.
            transaction.discard()
            if transaction.claimed:
                # Once response.create has made this work public, preserve normal
                # response semantics even if an abandoned speculative worker is
                # still waiting for an uncooperative provider.
                api_response = request()
                try:
                    yield from event_iterator(api_response)
                finally:
                    self._close_response(api_response)
            else:
                logger.warning("Skipping response prefetch while a previous provider worker is still active")
            return

        try:
            while not self._turn_is_cancelled(turn):
                try:
                    succeeded, value = results.get(timeout=0.05)
                except Empty:
                    continue
                if not succeeded:
                    worker.join()
                    raise value
                if value is done:
                    # Do not expose completion until the worker releases the
                    # sole provider slot; the next prefetch can then start
                    # without another scheduling-sensitive handoff.
                    worker.join()
                    return
                if self._turn_is_cancelled(turn):
                    break
                yield value
        finally:
            stop_reader.set()
            with response_lock:
                api_response = connected_response[0] if connected_response else None
            self._close_response(api_response)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.wait_for_gate(turn_id, turn_revision)

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
        *,
        language_name: str | None = None,
    ) -> None:
        if not instructions and not language_name:
            return
        builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
        full_instructions = builder(instructions or "", language_name=language_name)
        chat.prepend_instructions(full_instructions)

    # ── output helpers ──────────────────────────────────────────────────────--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
        history: ResponseHistory | None = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            history=history,
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
            selected_language=turn.selected_language,
            tools=tools or [],
            runtime_config=turn.runtime_config,
            response=turn.response,
            turn_id=turn.turn_id,
            turn_revision=turn.turn_revision,
            speech_stopped_at_s=turn.speech_stopped_at_s,
            cancel_generation=turn.gen,
            response_key=turn.response_key,
            prefetch_transaction=turn.prefetch_transaction,
        )

    def _record_tool_call(self, state: _GenState, turn: _Turn, event: ToolCall) -> Iterator[LLMOut]:
        """Emit a tool call with the history the service writes before exposing it.

        A fast client can return ``function_call_output`` before the response
        ends. Unless the call (and any text before it) is already in history,
        that output is rejected and the model repeats the call."""
        item = event.item
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if event.response_item is not None:
            fc_item = ResponsesFunctionCall(
                **fc_item.model_dump(exclude_unset=True), response_item=event.response_item.model_copy(deep=True)
            )
        if self._turn_is_cancelled(turn) or not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        state.history_items.append(fc_item)
        history = (
            None
            if is_out_of_band(turn.response)
            else ResponseHistory.capture(
                turn.runtime_config.chat,
                state.history_items,
                after_item_id=turn.history_anchor_id,
            )
        )
        state.output_emitted = True
        yield self._chunk(turn, tools=[item], history=history)

    # ── consumption ─────────────────────────────────────────────────────────--

    def _consume_streaming(
        self,
        events: Iterator[ProviderEvent],
        state: _GenState,
        turn: _Turn,
    ) -> Generator[LLMOut, None, bool]:
        cancelled = False
        printable_text = ""
        sentence_batch: list[str] = []

        def _flush(batch: list[str]) -> Iterator[LLMOut]:
            if not batch:
                return
            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            state.output_emitted = True
            yield self._chunk(turn, text=" ".join(batch))

        for event in events:
            # Provider usage is billable even when cancellation rolls back the
            # assistant output that accompanied it.
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
                continue
            if self._turn_is_cancelled(turn) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break

            if isinstance(event, ProviderResponseEnd):
                state.ending = event
                if event.status == "failed":
                    raise RuntimeError(event.error or "The language model provider reported a failed response.")
            elif isinstance(event, AssistantMessage):
                state.history_items.append(event.to_chat_item())
            elif isinstance(event, ResponseReasoningItem):
                state.history_items.append(event.model_copy(deep=True))
            elif isinstance(event, ToolCall):
                if state.ending.status != "completed":
                    continue
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(remove_markdown(printable_text.strip()))
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                yield from self._record_tool_call(state, turn, event)
            elif isinstance(event, TextDelta):
                if not turn.wants_audio:
                    # Text-only: forward verbatim. Keep every character (no
                    # remove_unspeechable, which strips TTS-unfriendly symbols) and
                    # don't sentence-split (sent_tokenize collapses newlines/markdown).
                    state.clean_text += event.text
                    if event.text:
                        if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                            logger.info("LLM generation cancelled (stale speculative turn)")
                            cancelled = True
                            break
                        state.output_emitted = True
                        yield self._chunk(turn, text=event.text)
                    continue
                new_text = remove_unspeechable(event.text)
                state.clean_text += new_text
                printable_text += new_text
                trailing_whitespace = printable_text[len(printable_text.rstrip()) :]
                sentences = sent_tokenize_preserving_markdown_code(printable_text, sent_tokenize)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(remove_markdown(s))
                        if len(sentence_batch) >= self.stream_batch_sentences:
                            if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                                logger.info("LLM generation cancelled (stale speculative turn)")
                                cancelled = True
                                break
                            yield from _flush(sentence_batch)
                            sentence_batch = []
                    if cancelled:
                        break
                    printable_text = sentences[-1] + trailing_whitespace

        if not cancelled:
            if printable_text.strip():
                sentence_batch.append(remove_markdown(printable_text.strip()))
            if sentence_batch:
                if self._turn_is_cancelled(turn):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug("Clean text: %s", transcript_for_log(state.clean_text))
                    yield from _flush(sentence_batch)
            logger.info("Tools: %s", transcript_for_log(state.tools))
        return (
            not cancelled
            and not self._turn_is_cancelled(turn)
            and self._turn_is_latest(turn.turn_id, turn.turn_revision)
            and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
        )

    def _consume_nonstreaming(
        self,
        events: Iterator[ProviderEvent],
        state: _GenState,
        turn: _Turn,
    ) -> Generator[LLMOut, None, bool]:
        cancelled = False
        for event in events:
            if isinstance(event, Usage):
                state.input_tokens = event.input_tokens
                state.output_tokens = event.output_tokens
                continue
            if self._turn_is_cancelled(turn) or not self._turn_is_latest(turn.turn_id, turn.turn_revision):
                logger.info("LLM generation cancelled (interruption)")
                cancelled = True
                break
            if isinstance(event, ProviderResponseEnd):
                state.ending = event
                if event.status == "failed":
                    raise RuntimeError(event.error or "The language model provider reported a failed response.")
            elif isinstance(event, AssistantMessage):
                state.history_items.append(event.to_chat_item())
            elif isinstance(event, ResponseReasoningItem):
                state.history_items.append(event.model_copy(deep=True))
            elif isinstance(event, ToolCall):
                if state.ending.status != "completed":
                    continue
                yield from self._record_tool_call(state, turn, event)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips markdown
                # and TTS-unfriendly symbols. Not per-delta here: each TextDelta
                # in the non-streaming path already carries the full response.
                spoken = event.text if not turn.wants_audio else remove_markdown(remove_unspeechable(event.text))
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._turn_is_cancelled(turn)
                    and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
                ):
                    state.output_emitted = True
                    yield self._chunk(turn, text=out)
        logger.debug("Clean text: %s", transcript_for_log(state.clean_text))
        logger.info("Tools: %s", transcript_for_log(state.tools))
        return (
            not cancelled
            and not self._turn_is_cancelled(turn)
            and self._turn_is_latest(turn.turn_id, turn.turn_revision)
            and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
        )

    # ── orchestration ─────────────────────────────────────────────────────────

    def _generate(
        self,
        active_chat: Chat,
        original_chat: Chat,
        turn: _Turn,
        optional_kwargs: dict[str, Any],
        *,
        serialize_fn: SerializeFn | None = None,
        request_fn: RequestFn | None = None,
        event_iterator_fn: EventIteratorFn | None = None,
        input_history: SupportedItem | None = None,
        audio_history_turns: int | None = None,
        consumed_audio_items: list[RealtimeConversationItemUserMessage] | None = None,
    ) -> Generator[LLMOut, None, bool]:
        api_response: Any = None
        events: Iterator[ProviderEvent] | None = None
        state = _GenState(history_items=[input_history] if input_history is not None else [])
        error_message: str | None = None
        generation_completed = False
        history = (
            None
            if is_out_of_band(turn.response)
            else ResponseHistory.capture(original_chat, [], after_item_id=turn.history_anchor_id)
        )
        provider_request_started = False
        consumed_image_ids: set[str] = set()
        store = getattr(self, "turn_latency_store", None)
        tracker = store.get_response(turn.response_key) if store else None

        try:
            generation_started_at_s = perf_counter()
            try:
                api_input = (serialize_fn or self._serialize)(active_chat)
                # Images the model actually sees this turn; only these are stripped on
                # write-back, so an image a fast client injects mid-generation for the
                # next turn survives (it is not in this serialized snapshot).
                consumed_image_ids = active_chat.image_message_ids()
                if not api_input:
                    # Nothing to send: empty `instructions` and no `input` (in the response,
                    # the default conversation, or the out-of-band context). The provider
                    # would reject this; fail with a clear message instead of an opaque error.
                    error_message = "Cannot generate a response: no instructions and no input were provided."
                else:
                    provider_request_started = True

                    def make_request() -> Any:
                        return (request_fn or self._request)(api_input, optional_kwargs)

                    if turn.prefetch_transaction is not None:
                        events = self._iter_prefetch_events_interruptibly(
                            make_request,
                            event_iterator_fn or self._iter_events,
                            turn,
                        )
                    else:
                        api_response = make_request()
                        events = (event_iterator_fn or self._iter_events)(api_response)
                if events is not None:

                    def measured_events() -> Iterator[ProviderEvent]:
                        for event in events:
                            if (
                                self.stream
                                and tracker is not None
                                and isinstance(event, TextDelta)
                                and event.text.strip()
                            ):
                                tracker.record_llm_ttft(perf_counter() - generation_started_at_s)
                            yield event

                    observed_events = measured_events()
                    if self.stream:
                        generation_completed = yield from self._consume_streaming(observed_events, state, turn)
                    else:
                        generation_completed = yield from self._consume_nonstreaming(observed_events, state, turn)
            except httpx.ReadTimeout:
                logger.warning(
                    "OpenAI API read timed out after %.1fs; ending the current response",
                    self.request_timeout_s,
                )
                error_message = f"Language model generation timed out after {self.request_timeout_s:.1f}s."
            except Exception as exc:
                # Any other generation failure must still terminate the response: record
                # the error and fall through to the EndOfResponse below. Without this the
                # exception would escape process() and no EndOfResponse would be emitted,
                # leaving st.in_response stuck and locking every subsequent response.
                log_exception(logger, "LLM generation failed; ending the current response", exc)
                if error_message is None:
                    error_message = f"Language model generation failed: {exc}"
            finally:
                # Store elapsed provider work before an error terminal can finish
                # the response on the service thread.
                if tracker is not None and provider_request_started:
                    tracker.record_llm(perf_counter() - generation_started_at_s)

            if (
                provider_request_started
                and error_message is not None
                and not state.output_emitted
                and (turn.prefetch_transaction is None or turn.prefetch_transaction.claimed)
                and not self._generation_is_stale(turn.gen)
                and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
            ):
                state.output_emitted = True
                yield LLMResponseChunk(
                    text=PROVIDER_FAILURE_FALLBACK,
                    selected_language=turn.selected_language,
                    runtime_config=turn.runtime_config,
                    response=turn.response,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    speech_stopped_at_s=turn.speech_stopped_at_s,
                    cancel_generation=turn.gen,
                    response_key=turn.response_key,
                    prefetch_transaction=turn.prefetch_transaction,
                )

            proposal_allowed = (
                error_message is None
                and state.ending.status == "completed"
                and generation_completed
                and not self._turn_is_cancelled(turn)
                and self._turn_is_latest(turn.turn_id, turn.turn_revision)
                and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
            )
            if proposal_allowed and not is_out_of_band(turn.response):
                history = ResponseHistory.capture(
                    original_chat,
                    state.history_items,
                    after_item_id=turn.history_anchor_id,
                    complete=True,
                    input_item_id=input_history.id if input_history is not None else None,
                    consumed_image_ids=consumed_image_ids,
                    item_order=state.ending.history_item_order,
                    audio_history_turns=audio_history_turns,
                    consumed_audio_items=consumed_audio_items,
                    compactor=self.compactor,
                )
            if turn.prefetch_transaction is not None and not proposal_allowed:
                turn.prefetch_transaction.discard()
            if state.input_tokens or state.output_tokens:
                yield TokenUsage(
                    input_tokens=state.input_tokens,
                    output_tokens=state.output_tokens,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    cancel_generation=turn.gen,
                    response_key=turn.response_key,
                )
            yield EndOfResponse(
                history=history,
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                cancel_generation=turn.gen,
                response_key=turn.response_key,
                error=error_message,
                input_tool_call_ids=(
                    active_chat.tool_output_call_ids() if proposal_allowed and not is_out_of_band(turn.response) else []
                ),
                status="incomplete" if generation_completed and state.ending.status == "incomplete" else "completed",
                reason=state.ending.reason if generation_completed and state.ending.status == "incomplete" else None,
            )
            return proposal_allowed
        finally:
            if turn.prefetch_transaction is not None and not generation_completed:
                # Publish failure to the shared transaction before the queued
                # logical-done event can race the client's response.create.
                turn.prefetch_transaction.discard()
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass

    def _process_audio(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process an audio-input turn through the selected backend protocol."""
        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        gen = self.cancel_scope.generation if self.cancel_scope else None
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=gen,
                response_key=request.response_key,
            )
            return

        original_chat = runtime_config.chat
        active_chat = original_chat.copy(deep=True)
        history_anchor_id = active_chat.history_anchor_id()
        if not is_out_of_band(response) and active_chat.has_pending_tool_calls():
            yield EndOfResponse(
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=gen,
                response_key=request.response_key,
                error="Cannot generate a response while function call outputs are pending.",
            )
            return
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(active_chat, response)
            except ChatItemError as exc:
                log_exception(logger, "Out-of-band response rejected", exc, level=logging.INFO)
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    response_key=request.response_key,
                    error=str(exc),
                )
                return

        language_code = request.language_code
        language_code, _ = resolve_auto_language(language_code)
        lang_name = language_name_for_prompt(language_code, enable=self.enable_lang_prompt)
        instructions = (
            response.instructions
            if response is not None and response.instructions is not None
            else runtime_config.session.instructions
        ) or ""
        req_tools = (
            response.tools if response is not None and response.tools is not None else runtime_config.session.tools
        )
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio, language_name=lang_name)

        optional_kwargs = self._build_audio_optional_kwargs(response, req_tools, req_tool_choice)
        input_history = None
        if request.audio is not None:
            audio_b64 = self._audio_to_wav_base64(request.audio, request.audio_sample_rate)
            audio_message = make_user_audio_message(audio_b64)
            if request.input_item_id is not None:
                audio_message.id = request.input_item_id
            active_chat.add_item(audio_message)
            if not is_out_of_band(response):
                input_history = audio_message

        # Cleanup only the audio this snapshot consumed. Audio that arrived or
        # was revised during generation must remain available for the next turn.
        consumed_audio_items = [
            item for item in active_chat.buffer if isinstance(item, RealtimeConversationItemUserMessage)
        ]

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        turn = _Turn(
            language_code=language_code,
            selected_language=request.selected_language,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
            response_key=request.response_key,
            prefetch_transaction=request.prefetch_transaction,
            history_anchor_id=history_anchor_id,
        )
        yield from self._generate(
            active_chat,
            original_chat,
            turn,
            optional_kwargs,
            serialize_fn=self._serialize_audio,
            request_fn=self._request_audio,
            event_iterator_fn=self._iter_audio_events,
            input_history=input_history,
            audio_history_turns=self.audio_history_turns,
            consumed_audio_items=consumed_audio_items,
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        retained_audio = not is_out_of_band(request.response) and any(
            part.type == "input_audio"
            for item in request.runtime_config.chat.copy().buffer
            if isinstance(item, RealtimeConversationItemUserMessage)
            for part in item.content
        )
        if request.audio is not None or retained_audio:
            yield from self._process_audio(request)
            return

        runtime_config = request.runtime_config
        response = request.response
        turn_id = request.turn_id
        turn_revision = request.turn_revision
        speech_stopped_at_s = request.speech_stopped_at_s
        gen = self.cancel_scope.generation if self.cancel_scope else None
        if not self._turn_is_latest(turn_id, turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", turn_id, turn_revision)
            yield EndOfResponse(
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=gen,
                response_key=request.response_key,
            )
            return

        original_chat = runtime_config.chat
        active_chat = original_chat.copy(deep=True)
        history_anchor_id = active_chat.history_anchor_id()
        if not is_out_of_band(response) and active_chat.has_pending_tool_calls():
            yield EndOfResponse(
                turn_id=turn_id,
                turn_revision=turn_revision,
                cancel_generation=gen,
                response_key=request.response_key,
                error="Cannot generate a response while function call outputs are pending.",
            )
            return
        if is_out_of_band(response):
            try:
                active_chat = build_active_chat(active_chat, response)
            except ChatItemError as exc:
                log_exception(logger, "Out-of-band response rejected", exc, level=logging.INFO)
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    response_key=request.response_key,
                    error=str(exc),
                )
                return
        language_code = request.language_code
        language_code, _ = resolve_auto_language(language_code)
        lang_name = language_name_for_prompt(language_code, enable=self.enable_lang_prompt)
        instructions = (
            response.instructions
            if response is not None and response.instructions is not None
            else runtime_config.session.instructions
        ) or ""
        req_tools = (
            response.tools if response is not None and response.tools is not None else runtime_config.session.tools
        )
        req_tool_choice = (
            response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        )
        wants_audio = response_wants_audio(response)
        self._apply_config(active_chat, instructions, wants_audio, language_name=lang_name)

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        turn = _Turn(
            language_code=language_code,
            selected_language=request.selected_language,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
            response_key=request.response_key,
            prefetch_transaction=request.prefetch_transaction,
            history_anchor_id=history_anchor_id,
        )
        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
