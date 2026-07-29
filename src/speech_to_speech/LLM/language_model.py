from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sized
from queue import Empty
from threading import Lock, Thread
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import torch
from nltk import sent_tokenize
from openai.types.realtime.realtime_conversation_item_function_call import (
    RealtimeConversationItemFunctionCall,
)
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams
from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline,
)

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.LLM.chat import (
    Chat,
    ChatItemError,
    build_active_chat,
    make_assistant_message,
    make_system_message,
    make_user_message,
)
from speech_to_speech.LLM.compaction_prompt import CompactGenerateFn, build_compactor
from speech_to_speech.LLM.text_prompt import build_text_system_prompt
from speech_to_speech.LLM.tool_call.function_call import extract_function_calls_from_text
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import END_CODE, ENTER_CODE, build_block_regex, build_tool_system_prompt
from speech_to_speech.LLM.utils import image_url_to_pil, remove_unspeechable, resolve_auto_language
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import LLMIn, LLMOut
from speech_to_speech.pipeline.messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.utils import is_out_of_band, response_wants_audio

try:
    import mlx.core as mx
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate as mlx_stream_generate

    from speech_to_speech.utils.mlx_lock import MLXLockContext

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from mlx_vlm import load as mlx_vlm_load
    from mlx_vlm import stream_generate as mlx_vlm_stream_generate

    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False

logger = logging.getLogger(__name__)


@runtime_checkable
class _Tokenizer(Protocol):
    """Minimal interface the base class needs from any tokenizer."""

    def encode(self, text: str, /, **kwargs: Any) -> list[int] | Sized: ...


@runtime_checkable
class _Processor(Protocol):
    """Minimal interface the VLM handler needs from any processor."""

    tokenizer: _Tokenizer

    def apply_chat_template(self, conversation: Any, /, **kwargs: Any) -> Any: ...
    def __call__(self, **kwargs: Any) -> Any: ...


class _CancelCriteria(StoppingCriteria):
    """Stopping criteria that can be signalled externally to abort generation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        return self._cancelled


class StreamContext(BaseModel):
    """Mutable accumulator passed through ``_stream_tokens`` so the caller
    can read back generation state after the iterator is exhausted."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cancelled: bool = False
    stopped: bool = False
    raw_generated_text: str = ""
    generated_text: str = ""
    printable_text: str = ""
    tools: list[ResponseFunctionToolCall] = Field(default_factory=list)
    function_tools: list[FunctionTool] = Field(default_factory=list)
    block_regex: Optional[str] = None
    enter_code: Optional[str] = None
    end_code: Optional[str] = None
    input_tokens: int = 0
    sentence_batch: list[str] = Field(default_factory=list)
    turn_id: str | None = None
    turn_revision: int | None = None
    speech_stopped_at_s: float | None = None
    cancel_generation: int | None = None

    @property
    def interrupted(self) -> bool:
        """True when generation ended early for any reason."""
        return self.cancelled or self.stopped


class BaseLanguageModelHandler(BaseHandler[LLMIn, LLMOut], ABC):
    """Abstract base for text-only and vision language model handlers.

    Holds shared pipeline logic (streaming, tool extraction, chat management)
    and delegates model-specific behaviour to two abstract hooks:
    ``_load_model`` and ``_generate``.
    """

    _cancel_criteria: _CancelCriteria
    streamer: Iterable[str]
    tokenizer: _Tokenizer

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        torch_dtype: str = "float16",
        gen_kwargs: dict[str, Any] = {},
        user_role: str = "user",
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        backend: Literal["transformers", "mlx"] = "transformers",
        enable_thinking: bool = False,
        stream_batch_sentences: int = 3,
        enable_lang_prompt: bool = False,
        compact_history: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.backend = backend
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.stream_batch_sentences = max(1, stream_batch_sentences)
        self.enable_lang_prompt = enable_lang_prompt

        self._load_model(model_name, device, torch_dtype, gen_kwargs)

        self.user_role = user_role
        # Serializes transformers pipe/model.generate calls between the speech
        # response path and the background compaction worker. MLX paths use
        # MLXLockContext instead.
        self._transformers_lock = Lock()
        self.compactor = build_compactor(self._build_compaction_generate_fn()) if compact_history else None

    def _turn_is_latest(self, turn_id: str | None, turn_revision: int | None) -> bool:
        return self.speculative_turns is None or self.speculative_turns.is_latest(turn_id, turn_revision)

    def _turn_output_allowed(self, turn_id: str | None, turn_revision: int | None) -> bool:
        if self.speculative_turns is None:
            return True
        return self.speculative_turns.is_latest_after_reopen_grace(turn_id, turn_revision)

    @abstractmethod
    def _load_model(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        gen_kwargs: dict[str, Any],
    ) -> None:
        """Load the model, tokenizer, and any backend-specific resources."""

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        """Return a ``(system, user) -> text`` callable for compaction.

        Subclasses must override this when ``compact_history=True`` is used.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _build_compaction_generate_fn. "
            "Override it or pass compact_history=False."
        )

    @abstractmethod
    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        """Run model generation, stream sentence chunks, set ``ctx.input_tokens``."""

    # ------------------------------------------------------------------
    # Generation lifecycle helpers
    # ------------------------------------------------------------------

    def _finish_mlx_generation(self, token_iter: Any) -> None:
        """Close the MLX generator to release resources immediately."""
        token_iter.close()

    def _finish_transformers_generation(self, thread: Thread) -> None:
        """Signal the stopping criteria, drain the streamer, and join the thread."""
        self._cancel_criteria.cancel()
        try:
            for _ in self.streamer:
                pass
        except Empty:
            pass
        thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _apply_instructions(
        self,
        chat: Chat,
        instructions: Optional[str],
        raw_tools: list[Any] | None,
        tool_choice: Optional[str],
        ctx: StreamContext | None = None,
        wants_audio: bool = True,
    ) -> None:
        if not instructions:
            return

        raw_tools = raw_tools or []

        function_tools = [FunctionTool(**t.model_dump()) for t in raw_tools if t.type == "function"]

        build_system_prompt = build_voice_system_prompt if wants_audio else build_text_system_prompt

        if function_tools and tool_choice != "none":
            tool_section = build_tool_system_prompt(function_tools, text_only=not wants_audio)
            full_instructions = build_system_prompt(instructions, tool_section=tool_section)
            block_regex = build_block_regex()
            enter_code = ENTER_CODE
            end_code = END_CODE
        else:
            full_instructions = build_system_prompt(instructions)
            block_regex = None
            enter_code = None
            end_code = None

        chat.add_item(make_system_message(full_instructions))

        if ctx is not None:
            ctx.function_tools = function_tools
            ctx.block_regex = block_regex
            ctx.enter_code = enter_code
            ctx.end_code = end_code

    def _process_printable_text(
        self,
        printable_text: str,
        language_code: Optional[str],
        tools: list[ResponseFunctionToolCall],
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> tuple[list[LLMResponseChunk], list[ResponseFunctionToolCall], str]:
        """Extract complete code blocks and return complete sentences to yield.

        Returns ``(chunks_to_yield, updated_tools, remaining_printable_text)``.
        Each element in *chunks_to_yield* is an :class:`LLMResponseChunk`
        ready to be yielded to the downstream pipeline.

        Sentences are accumulated in ``ctx.sentence_batch`` and only yielded
        when the batch reaches ``self.stream_batch_sentences``.
        """
        chunks: list[LLMResponseChunk] = []

        if ctx.enter_code and ctx.enter_code in printable_text:
            idx = printable_text.index(ctx.enter_code)
            before = printable_text[:idx]
            code_and_after = printable_text[idx:]
            if before.strip():
                for s in sent_tokenize(before):
                    ctx.sentence_batch.append(s)
            if ctx.sentence_batch:
                chunks.append(
                    LLMResponseChunk(
                        text=" ".join(ctx.sentence_batch),
                        language_code=language_code,
                        runtime_config=runtime_config,
                        response=response,
                        turn_id=ctx.turn_id,
                        turn_revision=ctx.turn_revision,
                        speech_stopped_at_s=ctx.speech_stopped_at_s,
                        cancel_generation=ctx.cancel_generation,
                    )
                )
                ctx.sentence_batch = []
            if ctx.block_regex and ctx.end_code and ctx.end_code in code_and_after:
                stripped, func_calls = extract_function_calls_from_text(
                    code_and_after,
                    ctx.block_regex,
                )
                parsed_tools: list[ResponseFunctionToolCall] = []
                for fc in func_calls:
                    if tools:
                        logger.warning(
                            "Skipping extra tool call '%s'; only one tool call is allowed per response",
                            fc.function_name,
                        )
                        continue
                    try:
                        tool_call = fc.to_realtime_function_tool_call(ctx.function_tools)
                    except ValueError as e:
                        logger.warning("Skipping invalid tool call: %s", e)
                        continue
                    tools.append(tool_call)
                    parsed_tools.append(tool_call)
                if parsed_tools:
                    chunks.append(
                        LLMResponseChunk(
                            text="",
                            language_code=language_code,
                            tools=parsed_tools,
                            runtime_config=runtime_config,
                            response=response,
                            turn_id=ctx.turn_id,
                            turn_revision=ctx.turn_revision,
                            speech_stopped_at_s=ctx.speech_stopped_at_s,
                            cancel_generation=ctx.cancel_generation,
                        )
                    )
                printable_text = stripped
            else:
                printable_text = code_and_after
            return chunks, tools, printable_text

        if printable_text and not response_wants_audio(response) and ctx.enter_code is None:
            # Text-only with no tool block: stream the raw text immediately. No TTS
            # means no need to split into sentences, and sent_tokenize would collapse
            # newlines / markdown. Streaming (vs buffering to the end) keeps the
            # response interruptible by a new speech turn.
            chunks.append(
                LLMResponseChunk(
                    text=printable_text,
                    language_code=language_code,
                    runtime_config=runtime_config,
                    response=response,
                    turn_id=ctx.turn_id,
                    turn_revision=ctx.turn_revision,
                    speech_stopped_at_s=ctx.speech_stopped_at_s,
                    cancel_generation=ctx.cancel_generation,
                )
            )
            return chunks, tools, ""

        if printable_text:
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    ctx.sentence_batch.append(s)
                    if len(ctx.sentence_batch) >= self.stream_batch_sentences:
                        chunks.append(
                            LLMResponseChunk(
                                text=" ".join(ctx.sentence_batch),
                                language_code=language_code,
                                runtime_config=runtime_config,
                                response=response,
                                turn_id=ctx.turn_id,
                                turn_revision=ctx.turn_revision,
                                speech_stopped_at_s=ctx.speech_stopped_at_s,
                                cancel_generation=ctx.cancel_generation,
                            )
                        )
                        ctx.sentence_batch = []
                printable_text = sentences[-1]

        return chunks, tools, printable_text

    def _check_stop(self, gen: int | None, ctx: StreamContext) -> bool:
        """Check whether generation should be aborted and mark the reason on *ctx*."""
        if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
            ctx.cancelled = True
            logger.info("LLM generation cancelled (interruption)")
            return True
        if not self._turn_is_latest(ctx.turn_id, ctx.turn_revision):
            ctx.cancelled = True
            logger.info("LLM generation cancelled (stale speculative turn)")
            return True
        if self.stop_event.is_set():
            ctx.stopped = True
            logger.info("LLM generation stopped (shutdown)")
            return True
        return False

    def _stream_tokens(
        self,
        token_iter: Iterator[Any],
        gen: int | None,
        language_code: Optional[str],
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        """Consume *token_iter*, accumulate text in *ctx*, yield sentence chunks."""
        # Text-only output keeps every character; only audio strips TTS-unfriendly
        # symbols via remove_unspeechable.
        wants_audio = response_wants_audio(response)
        while True:
            try:
                token = next(token_iter)
            except StopIteration:
                break
            except Empty:
                if self._check_stop(gen, ctx):
                    break
                continue

            if self._check_stop(gen, ctx):
                break

            raw_text: str = token.text if hasattr(token, "text") else token
            ctx.raw_generated_text += raw_text
            clean = raw_text if not wants_audio else remove_unspeechable(raw_text)
            ctx.generated_text += clean
            ctx.printable_text += clean
            chunks, ctx.tools, ctx.printable_text = self._process_printable_text(
                ctx.printable_text,
                language_code,
                ctx.tools,
                ctx,
                runtime_config,
                response,
            )
            if chunks and not self._turn_output_allowed(ctx.turn_id, ctx.turn_revision):
                ctx.cancelled = True
                logger.info("LLM generation cancelled (stale speculative turn)")
                break
            yield from chunks

        if ctx.sentence_batch and not ctx.interrupted:
            if ctx.printable_text.strip():
                ctx.sentence_batch.append(ctx.printable_text.strip())
                ctx.printable_text = ""
            if not self._turn_output_allowed(ctx.turn_id, ctx.turn_revision):
                ctx.cancelled = True
                logger.info("LLM generation cancelled (stale speculative turn)")
                return
            yield LLMResponseChunk(
                text=" ".join(ctx.sentence_batch),
                language_code=language_code,
                runtime_config=runtime_config,
                response=response,
                turn_id=ctx.turn_id,
                turn_revision=ctx.turn_revision,
                speech_stopped_at_s=ctx.speech_stopped_at_s,
                cancel_generation=ctx.cancel_generation,
            )
            ctx.sentence_batch = []

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def process(self, request: LLMIn) -> Iterator[LLMOut]:
        ctx = StreamContext()

        if not isinstance(request, GenerateResponseRequest):
            raise TypeError(f"Unexpected request type: {type(request)}")

        ctx.turn_id = request.turn_id
        ctx.turn_revision = request.turn_revision
        ctx.speech_stopped_at_s = request.speech_stopped_at_s
        if not self._turn_is_latest(ctx.turn_id, ctx.turn_revision):
            logger.info("Skipping stale LLM request for turn=%s rev=%s", ctx.turn_id, ctx.turn_revision)
            yield EndOfResponse(turn_id=ctx.turn_id, turn_revision=ctx.turn_revision)
            return

        runtime_config = request.runtime_config
        response = request.response
        original_chat = runtime_config.chat
        out_of_band = is_out_of_band(response)
        if out_of_band:
            try:
                active_chat = build_active_chat(original_chat, response)
            except ChatItemError as exc:
                logger.info("Out-of-band response rejected: %s", exc)
                yield EndOfResponse(turn_id=ctx.turn_id, turn_revision=ctx.turn_revision, error=str(exc))
                return
        else:
            active_chat = original_chat.copy()
        language_code = request.language_code
        instructions = (
            response.instructions if response and response.instructions else runtime_config.session.instructions
        )
        tools = response.tools if response and response.tools else runtime_config.session.tools
        tool_choice = response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
        self._apply_instructions(
            active_chat,
            instructions,
            tools,
            str(tool_choice) if tool_choice else None,
            ctx,
            response_wants_audio(response),
        )
        language_code, lang_name = resolve_auto_language(language_code)
        if lang_name and self.enable_lang_prompt:
            active_chat.add_item(make_user_message(f"Please reply to my message in {lang_name}."))

        gen = self.cancel_scope.generation if self.cancel_scope else None
        ctx.cancel_generation = gen
        # Images the model sees this turn; only these are stripped on write-back,
        # so an image a fast client injects mid-generation for the next turn
        # survives (it is not in this serialized snapshot).
        consumed_image_ids = active_chat.image_message_ids()

        try:
            yield from self._generate(active_chat, language_code, gen, ctx, runtime_config, response)

            if ctx.stopped:
                return

            turn_output_allowed = not ctx.cancelled and self._turn_output_allowed(ctx.turn_id, ctx.turn_revision)
            # Out-of-band responses still emit output, but never write back to the default
            # conversation (their context was a throwaway chat).
            commit_allowed = turn_output_allowed and not out_of_band
            if commit_allowed:
                original_chat.add_item(make_assistant_message(ctx.generated_text))
            if commit_allowed and ctx.tools:
                for t in ctx.tools:
                    original_chat.add_item(
                        RealtimeConversationItemFunctionCall(
                            type="function_call",
                            id=t.id,
                            call_id=t.call_id,
                            name=t.name,
                            arguments=t.arguments,
                            status=t.status,
                        )
                    )
            if commit_allowed:
                original_chat.strip_images(consumed_image_ids)
                original_chat.trim_if_needed(self.compactor)
            logger.debug("Clean text: %s", ctx.generated_text)
            logger.info(f"Tools: {ctx.tools}")

            if turn_output_allowed and ctx.printable_text.strip():
                yield LLMResponseChunk(
                    text=ctx.printable_text.strip(),
                    language_code=language_code,
                    runtime_config=runtime_config,
                    response=response,
                    turn_id=ctx.turn_id,
                    turn_revision=ctx.turn_revision,
                    speech_stopped_at_s=ctx.speech_stopped_at_s,
                    cancel_generation=ctx.cancel_generation,
                )

            output_tokens = len(self.tokenizer.encode(ctx.raw_generated_text)) if ctx.raw_generated_text else 0
            if turn_output_allowed and (ctx.input_tokens or output_tokens):
                yield TokenUsage(
                    input_tokens=ctx.input_tokens,
                    output_tokens=output_tokens,
                    turn_id=ctx.turn_id,
                    turn_revision=ctx.turn_revision,
                )
        except Exception as exc:
            # Any generation failure must still terminate the response. Without this
            # the exception would escape process() and no EndOfResponse would be
            # emitted, leaving st.in_response stuck and locking every later response.
            logger.exception("LLM generation failed; ending the current response")
            yield EndOfResponse(
                turn_id=ctx.turn_id,
                turn_revision=ctx.turn_revision,
                cancel_generation=ctx.cancel_generation,
                error=f"Language model generation failed: {exc}",
            )
            return
        yield EndOfResponse(
            turn_id=ctx.turn_id,
            turn_revision=ctx.turn_revision,
            cancel_generation=ctx.cancel_generation,
        )

    def on_session_end(self) -> None:
        logger.debug("Language model session state reset")


# ======================================================================
# Text-only LLM
# ======================================================================


class LanguageModelHandler(BaseLanguageModelHandler):
    """Text-only language model handler (transformers or MLX backend)."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    gen_kwargs: dict
    torch_dtype: torch.dtype
    pipe: Pipeline
    streamer: TextIteratorStreamer

    def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().setup(**kwargs)
        logger.info(f"LLM Backend: {self.backend}")
        self.warmup()

    def _load_model(self, model_name: str, device: str, torch_dtype: str, gen_kwargs: dict[str, Any]) -> None:

        logger.info("LLM Language Model Handler setup")

        if self.backend == "mlx":
            if not HAS_MLX:
                raise ImportError("mlx_lm is required for the 'mlx' backend. Install with: pip install mlx-lm")
            self.model, self.tokenizer = mlx_load(model_name)  # type: ignore[assignment, misc]
            self.gen_kwargs = gen_kwargs
        else:
            self.torch_dtype = getattr(torch, torch_dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[assignment]
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)  # type: ignore[arg-type]
            self.pipe = pipeline(  # type: ignore[call-overload]
                "text-generation", model=self.model, tokenizer=self.tokenizer, device=device
            )
            self.streamer = TextIteratorStreamer(
                self.tokenizer,  # type: ignore[arg-type]
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=1.0,
            )
            self._cancel_criteria = _CancelCriteria()
            self.gen_kwargs = {
                "streamer": self.streamer,
                "return_full_text": False,
                "stopping_criteria": StoppingCriteriaList([self._cancel_criteria]),
                **gen_kwargs,
            }

    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        chat_messages = chat.to_transformers_chat()
        chat_input = self.tokenizer.apply_chat_template(chat_messages, tokenize=True)
        ctx.input_tokens += len(chat_input if isinstance(chat_input, list) else chat_input["input_ids"])  # type: ignore[index]
        logger.debug("Prompt token count: %d", ctx.input_tokens)

        chat_prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        if self.backend == "mlx":
            with MLXLockContext(handler_name="MLX-LLM", timeout=10.0):
                token_iter = mlx_stream_generate(
                    self.model,  # type: ignore[arg-type]
                    self.tokenizer,  # type: ignore[arg-type]
                    chat_prompt,  # type: ignore[arg-type]
                    max_tokens=self.gen_kwargs["max_new_tokens"],
                )
                yield from self._stream_tokens(token_iter, gen, language_code, ctx, runtime_config, response)
                self._finish_mlx_generation(token_iter)
            try:
                mx.clear_cache()
            except Exception:
                pass
            torch.mps.empty_cache()
        else:
            self._cancel_criteria.reset()
            lock = self._transformers_lock

            def _locked_pipe() -> None:
                with lock:
                    self.pipe(chat_prompt, **self.gen_kwargs)

            thread = Thread(target=_locked_pipe)
            thread.start()
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx, runtime_config, response)
            self._finish_transformers_generation(thread)
            if self.device == "mps":
                torch.mps.empty_cache()

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        if self.backend == "mlx":
            model = self.model
            tokenizer = self.tokenizer
            max_tokens = 1024

            def generate_mlx(system: str, user: str) -> str:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    messages, tokenize=False, add_generation_prompt=True
                )
                with MLXLockContext(handler_name="MLX-compact", timeout=10.0):
                    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)  # type: ignore[arg-type]

            return generate_mlx
        else:
            pipe = self.pipe
            tokenizer = self.tokenizer
            gen_kwargs = {
                k: v
                for k, v in self.gen_kwargs.items()
                if k not in ("streamer", "stopping_criteria", "max_new_tokens", "return_full_text")
            }
            max_new_tokens = 1024
            lock = self._transformers_lock

            def generate_transformers(system: str, user: str) -> str:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                with lock:
                    result = pipe(prompt, return_full_text=False, max_new_tokens=max_new_tokens, **gen_kwargs)
                return result[0]["generated_text"]  # type: ignore[index]

            return generate_transformers

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        n_steps = 2

        if self.backend == "mlx":
            for _ in range(n_steps):
                prompt = self.tokenizer.apply_chat_template(dummy_chat, tokenize=False)
                mlx_generate(
                    self.model,  # type: ignore[arg-type]
                    self.tokenizer,  # type: ignore[arg-type]
                    prompt=prompt,  # type: ignore[arg-type]
                    max_tokens=self.gen_kwargs["max_new_tokens"],
                    verbose=False,
                )
        else:
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["min_new_tokens"],
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs,
            }

            if self.device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()

            for _ in range(n_steps):
                thread = Thread(target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs)
                thread.start()
                for _ in self.streamer:
                    pass

            if self.device == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                logger.info(
                    f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
                )


# ======================================================================
# Vision Language Model (VLM)
# ======================================================================


class VisionLanguageModelHandler(BaseLanguageModelHandler):
    """Vision language model handler (transformers or MLX-VLM backend)."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    processor: _Processor
    gen_kwargs: dict
    torch_dtype: torch.dtype
    streamer: TextIteratorStreamer

    def setup(self, **kwargs: Any) -> None:  # type: ignore[override]
        super().setup(**kwargs)
        logger.info(f"VLM Backend: {self.backend}")

    def _load_model(self, model_name: str, device: str, torch_dtype: str, gen_kwargs: dict[str, Any]) -> None:

        logger.info("VLM Language Model Handler setup")

        if self.backend == "mlx":
            if not HAS_MLX_VLM:
                raise ImportError("mlx-vlm is required for MLX VLM models. Install with: pip install mlx-vlm")
            self.model, self.processor = mlx_vlm_load(model_name)  # type: ignore[assignment]
            self.tokenizer = self.processor.tokenizer  # type: ignore[assignment]
            self.gen_kwargs = gen_kwargs
        else:
            self.torch_dtype = getattr(torch, torch_dtype)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)  # type: ignore[assignment]
            self.tokenizer = self.processor.tokenizer  # type: ignore[assignment]
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
            ).to(device)  # type: ignore[arg-type]
            self.streamer = TextIteratorStreamer(
                self.tokenizer,  # type: ignore[arg-type]
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=1.0,
            )
            self._cancel_criteria = _CancelCriteria()
            self.gen_kwargs = gen_kwargs

    def _generate(
        self,
        chat: Chat,
        language_code: Optional[str],
        gen: int | None,
        ctx: StreamContext,
        runtime_config: RuntimeConfig | None = None,
        response: RealtimeResponseCreateParams | None = None,
    ) -> Iterator[LLMResponseChunk]:
        prepared = chat.to_transformers_chat()
        if self.backend == "mlx":
            images, formatted_prompt = self._prepare_mlx_vlm_inputs(prepared)
            ctx.input_tokens += len(self.tokenizer.encode(formatted_prompt))
            logger.debug("MLX VLM prompt token count: %d", ctx.input_tokens)

            with MLXLockContext(handler_name="MLX-VLM", timeout=10.0):
                token_iter = mlx_vlm_stream_generate(  # type: ignore[arg-type]
                    self.model,
                    self.processor,
                    formatted_prompt,
                    images or None,
                    max_tokens=self.gen_kwargs.get("max_new_tokens", 1024),
                    enable_thinking=self.enable_thinking,
                )
                yield from self._stream_tokens(token_iter, gen, language_code, ctx, runtime_config, response)
                self._finish_mlx_generation(token_iter)
            try:
                mx.clear_cache()
            except Exception:
                pass
            torch.mps.empty_cache()
        else:
            inputs, input_tokens = self._prepare_vlm_inputs(prepared)
            ctx.input_tokens += input_tokens
            logger.debug("VLM prompt token count: %d", ctx.input_tokens)

            self._cancel_criteria.reset()
            generate_kwargs = {
                **inputs,
                "max_new_tokens": self.gen_kwargs.get("max_new_tokens", 1024),
                "streamer": self.streamer,
                "stopping_criteria": StoppingCriteriaList([self._cancel_criteria]),
            }
            lock = self._transformers_lock

            def _locked_generate() -> None:
                with lock:
                    self.model.generate(**generate_kwargs)  # type: ignore[union-attr,operator]

            thread = Thread(target=_locked_generate)
            thread.start()
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx, runtime_config, response)
            self._finish_transformers_generation(thread)
            if self.device == "mps":
                torch.mps.empty_cache()

    def _build_compaction_generate_fn(self) -> CompactGenerateFn:
        if self.backend == "mlx":
            model = self.model
            processor = self.processor
            tokenizer = self.tokenizer
            max_tokens = 1024

            def generate_mlx(system: str, user: str) -> str:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                formatted_prompt = processor.apply_chat_template(  # type: ignore[union-attr]
                    messages, tokenize=False, add_generation_prompt=True
                )
                with MLXLockContext(handler_name="MLX-VLM-compact", timeout=10.0):
                    token_iter = mlx_vlm_stream_generate(  # type: ignore[arg-type]
                        model, processor, formatted_prompt, None, max_tokens=max_tokens
                    )
                    return "".join(t.text if hasattr(t, "text") else str(t) for t in token_iter)

            return generate_mlx
        else:
            model = self.model
            processor = self.processor
            tokenizer = self.tokenizer
            device = self.device
            max_new_tokens = 1024
            lock = self._transformers_lock

            def generate_transformers(system: str, user: str) -> str:
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                prompt = processor.apply_chat_template(  # type: ignore[union-attr]
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)  # type: ignore[operator]
                input_len = inputs["input_ids"].shape[1]
                with lock:
                    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[union-attr,operator]
                return str(tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True))  # type: ignore[union-attr]

            return generate_transformers

    def _prepare_vlm_inputs(self, chat_messages: list[dict[str, Any]]) -> tuple[Any, int]:
        """Build processor inputs for transformers VLM generation.

        Converts Realtime-style content (``input_text``/``input_image``) to the
        transformers chat-template format (``text``/``image``) and decodes image
        URLs to PIL images.
        """
        converted_messages = []
        images = []
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "input_text":
                        parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "input_image":
                        pil_img = image_url_to_pil(p["image_url"])
                        images.append(pil_img)
                        parts.append({"type": "image", "image": pil_img})
                converted_messages.append({"role": msg["role"], "content": parts})
            else:
                converted_messages.append(msg)

        text_prompt = self.processor.apply_chat_template(
            converted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        return inputs, len(inputs["input_ids"][0])

    def _prepare_mlx_vlm_inputs(self, chat_messages: list[dict[str, Any]]) -> tuple[list[Any], str]:
        """Build formatted prompt and extract PIL images for MLX VLM generation.

        Converts Realtime-style content (``input_text``/``input_image``) to the
        HF chat-template format and uses the processor's own ``apply_chat_template``
        for full multi-turn support.

        Returns ``(images, formatted_prompt)`` ready for ``mlx_vlm.stream_generate``.
        """
        images = []
        converted_messages = []
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "input_text":
                        parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "input_image":
                        images.append(image_url_to_pil(p["image_url"]))
                        parts.append({"type": "image"})
                converted_messages.append({"role": msg["role"], "content": parts})
            else:
                converted_messages.append(msg)

        formatted_prompt = self.processor.apply_chat_template(
            converted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return images, formatted_prompt
