from __future__ import annotations

import base64
import io
import logging
import os
import wave
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from queue import Empty, Full, Queue
from threading import BoundedSemaphore, Lock, Thread, current_thread
from threading import Event as ThreadingEvent
from typing import Any, Literal, Optional

import httpx
import numpy as np
from nltk import sent_tokenize
from openai import OpenAI
from openai.types.realtime.conversation_item import (
    RealtimeConversationItemAssistantMessage,
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_conversation_item_assistant_message import (
    Content as AssistantContent,
)
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.config.providers import (
    DEFAULT_PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
    detect_provider,
    is_local_base_url,
    is_official_openai,
    resolve_credentials,
)

__all__ = [
    "BaseOpenAICompatibleHandler",
    "DEFAULT_PROVIDER_BASE_URLS",
    "PROVIDER_ENV_KEYS",
]
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    SupportedItem,
    build_active_chat,
    make_system_message,
    make_user_audio_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.utils import remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    LLMResponseChunk,
    ResponsePrefetchTransaction,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

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


class ToolCall(BaseModel):
    """A complete function tool call (``call_id`` / ``id`` already regenerated)."""

    item: ResponseFunctionToolCall


class Usage(BaseModel):
    """Token accounting for the turn."""

    input_tokens: int
    output_tokens: int


ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage
SerializeFn = Callable[[Chat], Any]
RequestFn = Callable[[Any, dict[str, Any]], Any]
EventIteratorFn = Callable[[Any], Iterator[ProviderEvent]]


class _Turn(BaseModel):
    """Per-request context threaded through generation (immutable for the turn)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    language_code: Optional[str]
    gen: int | None
    runtime_config: Any
    response: Any
    turn_id: str | None
    turn_revision: int | None
    speech_stopped_at_s: float | None
    wants_audio: bool
    response_key: str
    prefetch_transaction: ResponsePrefetchTransaction | None = None


class _GenState(BaseModel):
    """Mutable accumulators collected while consuming a turn's events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    pending: list[SupportedItem] = Field(default_factory=list)
    recorded_item_ids: set[str] = Field(default_factory=set)
    recorded_call_ids: set[str] = Field(default_factory=set)
    clean_text: str = ""  # filtered text, kept only for the debug log
    input_tokens: int = 0
    output_tokens: int = 0
    output_emitted: bool = False



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

    @classmethod
    def _detect_provider(cls, model_name: Optional[str], base_url: Optional[str]) -> Optional[str]:
        """Detect provider name based on model name or base URL."""
        spec = detect_provider(model_name, base_url)
        return spec.name if spec else None

    @classmethod
    def resolve_provider_credentials(
        cls,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve base_url and api_key from CLI arguments, .env environment variables, and defaults."""
        return resolve_credentials(model_name=model_name, base_url=base_url, api_key=api_key)

    # ── setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        model_name: str = "gpt-5.4-mini",
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
        env_model = os.environ.get("MODEL_NAME") or os.environ.get("LLM_MODEL_NAME")
        if env_model and (model_name == "gpt-5.4-mini" or model_name is None):
            model_name = env_model
        elif model_name is None:
            model_name = "gpt-5.4-mini"
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
        self.request_timeout_s = float(request_timeout_s)
        self.request_timeout = httpx.Timeout(
            self.request_timeout_s,
            connect=min(10.0, self.request_timeout_s),
        )

        self.user_role = user_role

        base_url, api_key = self.resolve_provider_credentials(
            model_name=self.model_name,
            base_url=base_url,
            api_key=api_key,
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._extra_body = self._build_extra_body(base_url, disable_thinking, reasoning_effort)
        self._prefetch_worker_slots = BoundedSemaphore(PREFETCH_PROVIDER_WORKER_LIMIT)
        self._prefetch_workers_lock = Lock()
        self._prefetch_workers: set[Thread] = set()
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None
        self.warmup()

    @staticmethod
    def _is_official_openai(base_url: Optional[str]) -> bool:
        """Whether base_url points at the official OpenAI server."""
        return is_official_openai(base_url)

    @staticmethod
    def _is_local_base_url(base_url: str) -> bool:
        """Whether base_url points at localhost or a loopback IP address."""
        return is_local_base_url(base_url)

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
        if base_url is None or cls._is_official_openai(base_url) or (base_url and "googleapis.com" in base_url):
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

    @staticmethod
    def _audio_to_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
        """Encode a mono 16-bit WAV payload without touching the filesystem."""
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
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    def _apply_config(
        self,
        chat: Chat,
        instructions: Optional[str],
        wants_audio: bool = True,
    ) -> None:
        if instructions:
            builder = build_voice_system_prompt if wants_audio else build_text_system_prompt
            full_instructions = builder(instructions)
            chat.add_item(make_system_message(full_instructions))

    # ── output helpers ──────────────────────────────────────────────────────--

    def _chunk(
        self,
        turn: _Turn,
        *,
        text: str = "",
        tools: list[ResponseFunctionToolCall] | None = None,
        language_code: Optional[str] = None,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            text=text,
            language_code=language_code if language_code is not None else turn.language_code,
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

    def _record_tool_call(self, state: _GenState, turn: _Turn, item: ResponseFunctionToolCall) -> Iterator[LLMOut]:
        """Emit a tool call, persisting it (and any assistant text seen so far)
        to history *before* it is forwarded to the client.

        The function_call must already exist in the conversation by the time the
        client returns its ``function_call_output``; otherwise a fast client
        races ahead of the deferred end-of-turn write-back and the output is
        rejected ("No function_call with call_id ... found"), which makes the
        model re-issue the same tool call. The call lands in ``_pending_tool_calls``
        at its emitted position (not serialized until its output pairs it), so
        eager recording is safe.

        Out-of-band turns never touch the default conversation, and a stale turn
        records nothing (it is not forwarded to the client either)."""
        state.tools.append(item)
        fc_item = RealtimeConversationItemFunctionCall(
            type="function_call",
            name=item.name,
            arguments=item.arguments,
            call_id=item.call_id,
            id=item.id,
            status=item.status,
        )
        if self._turn_is_cancelled(turn) or not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
            logger.info("LLM generation cancelled (stale speculative turn)")
            return
        if not is_out_of_band(turn.response):
            # Flush assistant text accumulated before this call first (so history
            # order matches what the client received), then persist the call —
            # all before the chunk leaves for the client.
            chat = turn.runtime_config.chat
            recorded_items = chat.add_provisional_generation_items(
                turn.response_key,
                [*state.pending, fc_item],
            )
            state.pending.clear()
            if recorded_items is None:
                logger.info("LLM generation cancelled before tool output was recorded")
                return
            for recorded in recorded_items:
                if recorded.id is not None:
                    state.recorded_item_ids.add(recorded.id)
                if isinstance(recorded, RealtimeConversationItemFunctionCall) and recorded.call_id is not None:
                    state.recorded_call_ids.add(recorded.call_id)
        state.output_emitted = True
        yield self._chunk(turn, tools=[item])

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

            if isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                # Flush any pending spoken text before emitting the tool call.
                if printable_text.strip():
                    sentence_batch.append(printable_text.strip())
                    printable_text = ""
                if sentence_batch:
                    if not self._turn_output_allowed(turn.turn_id, turn.turn_revision):
                        logger.info("LLM generation cancelled (stale speculative turn)")
                        cancelled = True
                        break
                    yield from _flush(sentence_batch)
                    sentence_batch = []
                yield from self._record_tool_call(state, turn, event.item)
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
                sentences = sent_tokenize(printable_text)
                if len(sentences) > 1:
                    for s in sentences[:-1]:
                        sentence_batch.append(s)
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
                sentence_batch.append(printable_text.strip())
            if sentence_batch:
                if self._turn_is_cancelled(turn):
                    logger.info("LLM generation cancelled (interruption)")
                else:
                    logger.debug(f"Clean text: {state.clean_text}")
                    yield from _flush(sentence_batch)
            logger.info(f"Tools: {state.tools}")
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
            if isinstance(event, AssistantMessage):
                state.pending.append(
                    RealtimeConversationItemAssistantMessage(type="message", role="assistant", content=event.content)
                )
            elif isinstance(event, ToolCall):
                yield from self._record_tool_call(state, turn, event.item)
            elif isinstance(event, TextDelta):
                # Text-only keeps every character verbatim; audio strips
                # TTS-unfriendly symbols via remove_unspeechable.
                spoken = event.text if not turn.wants_audio else remove_unspeechable(event.text)
                state.clean_text += spoken
                out = spoken if not turn.wants_audio else spoken.strip()
                if (
                    out
                    and not self._turn_is_cancelled(turn)
                    and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
                ):
                    state.output_emitted = True
                    yield self._chunk(turn, text=out)
        logger.debug(f"Clean text: {state.clean_text}")
        logger.info(f"Tools: {state.tools}")
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
        transactional_user_message_id: str | None = None,
        history_commit_fn: Callable[[], None] | None = None,
    ) -> Generator[LLMOut, None, bool]:
        api_response: Any = None
        events: Iterator[ProviderEvent] | None = None
        state = _GenState()
        error_message: str | None = None
        generation_completed = False
        history_committed = False
        transaction_rolled_back = False
        provider_request_started = False
        consumed_image_ids: set[str] = set()

        def rollback_transaction() -> None:
            nonlocal transaction_rolled_back
            if history_committed or transaction_rolled_back:
                return
            if transactional_user_message_id is None and not (state.recorded_item_ids or state.recorded_call_ids):
                return
            original_chat.rollback_generation(
                transactional_user_message_id,
                item_ids=state.recorded_item_ids,
                call_ids=state.recorded_call_ids,
                response_key=turn.response_key,
            )
            transaction_rolled_back = True

        try:
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
                    if self.stream:
                        generation_completed = yield from self._consume_streaming(events, state, turn)
                    else:
                        generation_completed = yield from self._consume_nonstreaming(events, state, turn)
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
                logger.exception("LLM generation failed; ending the current response")
                if error_message is None:
                    error_message = f"Language model generation failed: {exc}"

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
                    runtime_config=turn.runtime_config,
                    response=turn.response,
                    turn_id=turn.turn_id,
                    turn_revision=turn.turn_revision,
                    speech_stopped_at_s=turn.speech_stopped_at_s,
                    cancel_generation=turn.gen,
                    response_key=turn.response_key,
                    prefetch_transaction=turn.prefetch_transaction,
                )

            can_commit = (
                error_message is None
                and generation_completed
                and not self._turn_is_cancelled(turn)
                and self._turn_is_latest(turn.turn_id, turn.turn_revision)
                and self._turn_output_allowed(turn.turn_id, turn.turn_revision)
            )
            if can_commit:
                try:
                    # Out-of-band responses emit output and usage but never write back to the
                    # default conversation (their context was a throwaway chat).
                    if not is_out_of_band(turn.response):
                        # Tool calls (and any assistant text preceding them) were already
                        # written eagerly in _record_tool_call; only trailing items remain.
                        recorded_items = original_chat.add_provisional_generation_items(
                            turn.response_key,
                            state.pending,
                            committed_item_ids=(
                                {transactional_user_message_id} if transactional_user_message_id is not None else None
                            ),
                        )
                        if recorded_items is None:
                            can_commit = False
                        for recorded in recorded_items or []:
                            if recorded.id is not None:
                                state.recorded_item_ids.add(recorded.id)
                        if can_commit:

                            def cleanup_history() -> None:
                                snapshot = original_chat.snapshot_history_cleanup()
                                try:
                                    original_chat.strip_images(consumed_image_ids)
                                    if history_commit_fn is not None:
                                        history_commit_fn()
                                    original_chat.trim_if_needed(self.compactor)
                                except Exception:
                                    original_chat.restore_history_cleanup(snapshot)
                                    raise

                            if turn.prefetch_transaction is not None:
                                turn.prefetch_transaction.complete(cleanup_history)
                            else:
                                cleanup_history()
                    history_committed = can_commit
                except Exception as exc:
                    logger.exception("LLM history commit failed; rolling back the current response")
                    error_message = f"Language model history commit failed: {exc}"

            rollback_transaction()
            if turn.prefetch_transaction is not None and not history_committed:
                # Mark hidden failure before yielding usage/terminal output;
                # consumers run concurrently between generator resumptions.
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
                turn_id=turn.turn_id,
                turn_revision=turn.turn_revision,
                cancel_generation=turn.gen,
                response_key=turn.response_key,
                error=error_message,
            )
            return history_committed
        finally:
            if turn.prefetch_transaction is not None and not history_committed:
                # Publish failure to the shared transaction before the queued
                # logical-done event can race the client's response.create.
                turn.prefetch_transaction.discard()
            if api_response is not None and hasattr(api_response, "close"):
                try:
                    api_response.close()
                except Exception:
                    pass
            rollback_transaction()

    def _process_audio(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process an audio-input turn through the selected backend protocol."""
        assert request.audio is not None
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
        if not is_out_of_band(response) and original_chat.has_pending_tool_calls():
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
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    response_key=request.response_key,
                    error=str(exc),
                )
                return
        else:
            active_chat = original_chat.copy()

        language_code = request.language_code
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
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        audio_b64 = self._audio_to_wav_base64(request.audio, request.audio_sample_rate)
        audio_message = active_chat.add_item(make_user_audio_message(audio_b64))
        optional_kwargs = self._build_audio_optional_kwargs(response, req_tools, req_tool_choice)

        transactional_user_message_id: str | None = None
        history_commit_fn: Callable[[], None] | None = None
        if not is_out_of_band(response):
            provisional_message = make_user_audio_message(audio_b64)
            provisional_message.id = audio_message.id
            recorded_items = original_chat.add_provisional_generation_items(
                request.response_key,
                [provisional_message],
            )
            if recorded_items is None:
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    response_key=request.response_key,
                )
                return
            assert provisional_message.id is not None
            transactional_user_message_id = provisional_message.id

            def commit_audio_history() -> None:
                original_chat.compact_audio_history(self.audio_history_turns)

            history_commit_fn = commit_audio_history

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
            response_key=request.response_key,
            prefetch_transaction=request.prefetch_transaction,
        )
        yield from self._generate(
            active_chat,
            original_chat,
            turn,
            optional_kwargs,
            serialize_fn=self._serialize_audio,
            request_fn=self._request_audio,
            event_iterator_fn=self._iter_audio_events,
            transactional_user_message_id=transactional_user_message_id,
            history_commit_fn=history_commit_fn,
        )

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        """Process a language model request and yield LLMResponseChunks."""
        if request.audio is not None:
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
        if not is_out_of_band(response) and original_chat.has_pending_tool_calls():
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
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                    cancel_generation=gen,
                    response_key=request.response_key,
                    error=str(exc),
                )
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
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
        self._apply_config(active_chat, instructions, wants_audio)
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        optional_kwargs = self._build_optional_kwargs(req_tools, req_tool_choice)

        # CancelScope.is_stale(gen) is checked when the stream iterator advances; a
        # blocked read inside httpx cannot be aborted by cancel_scope.cancel() from
        # the websocket router. Mitigations: request_timeout_s / ReadTimeout.
        turn = _Turn(
            language_code=language_code,
            gen=gen,
            runtime_config=runtime_config,
            response=response,
            turn_id=turn_id,
            turn_revision=turn_revision,
            speech_stopped_at_s=speech_stopped_at_s,
            wants_audio=wants_audio,
            response_key=request.response_key,
            prefetch_transaction=request.prefetch_transaction,
        )
        yield from self._generate(active_chat, original_chat, turn, optional_kwargs)

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def should_log_timing(self, output: LLMOut) -> bool:
        return isinstance(output, LLMResponseChunk) and self.last_time > self.min_time_to_debug
