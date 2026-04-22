from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
    TextIteratorStreamer,
)
import json
import torch
import logging
from queue import Empty
from nltk import sent_tokenize
from pydantic import BaseModel, ConfigDict, Field

from LLM.chat import Chat
from LLM.tool_call.function_call import extract_function_calls_from_text
from LLM.tool_call.function_tool import FunctionTool
from openai.types.responses import ResponseFunctionToolCall
from LLM.tool_call.tool_prompt import build_tool_system_prompt, build_block_regex, ENTER_CODE, END_CODE
from typing import Any, Literal
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from LLM.utils import remove_unspeechable, resolve_auto_language, image_url_to_pil
from LLM.voice_prompt import build_voice_system_prompt
from pipeline_messages import (
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
    Transcription,
)

try:
    from mlx_lm import load as mlx_load, stream_generate as mlx_stream_generate, generate as mlx_generate
    from utils.mlx_lock import MLXLockContext
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from mlx_vlm import load as mlx_vlm_load, stream_generate as mlx_vlm_stream_generate
    HAS_MLX_VLM = True
except ImportError:
    HAS_MLX_VLM = False

logger = logging.getLogger(__name__)


class _CancelCriteria(StoppingCriteria):
    """Stopping criteria that can be signalled externally to abort generation."""

    def __init__(self):
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
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
    block_regex: str | None = None
    enter_code: str | None = None
    end_code: str | None = None
    input_tokens: int = 0
    sentence_batch: list[str] = Field(default_factory=list)

    @property
    def interrupted(self) -> bool:
        """True when generation ended early for any reason."""
        return self.cancelled or self.stopped


class BaseLanguageModelHandler(BaseHandler[Transcription | GenerateResponseRequest], ABC):
    """Abstract base for text-only and vision language model handlers.

    Holds shared pipeline logic (streaming, tool extraction, chat management)
    and delegates model-specific behaviour to two abstract hooks:
    ``_load_model`` and ``_generate``.
    """

    def setup(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "cuda",
        torch_dtype: str = "float16",
        gen_kwargs: dict = {},
        user_role: str = "user",
        chat_size: int = 1,
        init_chat_role: str | None = None,
        init_chat_prompt: str = "You are a helpful AI assistant.",
        cancel_scope: CancelScope | None = None,
        backend: Literal["transformers", "mlx"] = "transformers",
        enable_thinking: bool = False,
        stream_batch_sentences: int = 3,
    ):
        self.backend = backend
        self.cancel_scope = cancel_scope
        self.device = device
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.stream_batch_sentences = max(1, stream_batch_sentences)

        self._load_model(model_name, device, torch_dtype, gen_kwargs)

        # TODO: chat is not used in the realtime pipeline, but still need to be kept for backward compatibility. Remove it in the future.
        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            full_prompt = build_voice_system_prompt(init_chat_prompt)
            self.chat.init_chat({"role": init_chat_role, "content": full_prompt})

        self.user_role = user_role

    @abstractmethod
    def _load_model(
        self,
        model_name: str,
        device: str,
        torch_dtype: str,
        gen_kwargs: dict,
    ) -> None:
        """Load the model, tokenizer, and any backend-specific resources."""

    @abstractmethod
    def _generate(
        self,
        chat: Chat,
        language_code: str | None,
        gen: int | None,
        ctx: StreamContext,
        runtime_config=None,
        response=None,
    ) -> Iterator[LLMResponseChunk]:
        """Run model generation, stream sentence chunks, set ``ctx.input_tokens``."""

    # ------------------------------------------------------------------
    # Generation lifecycle helpers
    # ------------------------------------------------------------------

    def _finish_mlx_generation(self, token_iter) -> None:
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

    @staticmethod
    def _prepare_chat_for_transformers(chat_messages: list[dict]) -> list[dict]:
        """Convert Responses-API-style tool items to the transformers chat format.

        - ``function_call_output`` items become ``{"role": "tool", ...}``
        - Messages with ``tool_calls`` and regular messages pass through as-is
        - ``function_call`` items (from the OpenAI Responses API path) are
          converted to assistant messages with ``tool_calls``
        """
        result = []
        for msg in chat_messages:
            item_type = msg.get("type")
            if item_type == "function_call_output":
                call_id = msg.get("call_id", "")
                name = ""
                for prev in reversed(result):
                    if prev.get("role") == "assistant" and "tool_calls" in prev:
                        for tc in prev["tool_calls"]:
                            if tc.get("id") == call_id:
                                name = tc.get("function", {}).get("name", "")
                                break
                        if name:
                            break
                result.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": msg.get("output", ""),
                })
            elif item_type == "function_call":
                args = msg.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                result.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "type": "function",
                        "id": msg.get("call_id", ""),
                        "function": {
                            "name": msg.get("name", ""),
                            "arguments": args,
                        },
                    }],
                })
            else:
                result.append(msg)
        return result

    def _apply_instructions(
        self,
        chat: Chat,
        instructions: str | None,
        raw_tools: list | None,
        tool_choice: str | None,
        ctx: StreamContext | None = None,
    ) -> None:
        if not instructions:
            return

        raw_tools = raw_tools or []

        function_tools = [
            FunctionTool(**t.model_dump())
            for t in raw_tools
            if t.type == "function"
        ]

        if function_tools and tool_choice != "none":
            tool_section = build_tool_system_prompt(function_tools)
            full_instructions = build_voice_system_prompt(instructions, tool_section=tool_section)
            block_regex = build_block_regex()
            enter_code = ENTER_CODE
            end_code = END_CODE
        else:
            full_instructions = build_voice_system_prompt(instructions)
            block_regex = None
            enter_code = None
            end_code = None

        chat.init_chat({"role": "system", "content": full_instructions})

        if ctx is not None:
            ctx.function_tools = function_tools
            ctx.block_regex = block_regex
            ctx.enter_code = enter_code
            ctx.end_code = end_code

    def _process_printable_text(
        self, printable_text: str, language_code: str | None, tools: list[ResponseFunctionToolCall],
        ctx: StreamContext, runtime_config=None, response=None,
    ) -> tuple[list[LLMResponseChunk], list[ResponseFunctionToolCall], str]:
        """Extract complete code blocks and return complete sentences to yield.

        Returns ``(chunks_to_yield, updated_tools, remaining_printable_text)``.
        Each element in *chunks_to_yield* is an :class:`LLMResponseChunk`
        ready to be yielded to the downstream pipeline.

        Sentences are accumulated in ``ctx.sentence_batch`` and only yielded
        when the batch reaches ``self.stream_batch_sentences``.
        """
        chunks: list[LLMResponseChunk] = []

        if ctx.block_regex and ctx.end_code and ctx.end_code in printable_text:
            stripped, func_calls = extract_function_calls_from_text(
                printable_text, ctx.block_regex,
            )
            for fc in func_calls:
                try:
                    tools.append(
                        fc.to_realtime_function_tool_call(ctx.function_tools)
                    )
                except ValueError as e:
                    logger.warning("Skipping invalid tool call: %s", e)
            printable_text = stripped

        if ctx.enter_code and ctx.enter_code in printable_text:
            idx = printable_text.index(ctx.enter_code)
            before = printable_text[:idx]
            if before.strip():
                for s in sent_tokenize(before):
                    ctx.sentence_batch.append(s)
                    if len(ctx.sentence_batch) >= self.stream_batch_sentences:
                        chunks.append(LLMResponseChunk(text=" ".join(ctx.sentence_batch), language_code=language_code, runtime_config=runtime_config, response=response))
                        ctx.sentence_batch = []
            printable_text = printable_text[idx:]
            return chunks, tools, printable_text

        if printable_text:
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    ctx.sentence_batch.append(s)
                    if len(ctx.sentence_batch) >= self.stream_batch_sentences:
                        chunks.append(LLMResponseChunk(text=" ".join(ctx.sentence_batch), language_code=language_code, runtime_config=runtime_config, response=response))
                        ctx.sentence_batch = []
                printable_text = sentences[-1]

        return chunks, tools, printable_text

    def _check_stop(self, gen: int | None, ctx: StreamContext) -> bool:
        """Check whether generation should be aborted and mark the reason on *ctx*."""
        if gen is not None and self.cancel_scope.is_stale(gen):
            ctx.cancelled = True
            logger.info("LLM generation cancelled (interruption)")
            return True
        if self.stop_event.is_set():
            ctx.stopped = True
            logger.info("LLM generation stopped (shutdown)")
            return True
        return False

    def _stream_tokens(
        self,
        token_iter: Iterator,
        gen: int | None,
        language_code: str | None,
        ctx: StreamContext,
        runtime_config=None,
        response=None,
    ) -> Iterator[LLMResponseChunk]:
        """Consume *token_iter*, accumulate text in *ctx*, yield sentence chunks."""
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
            clean = remove_unspeechable(raw_text)
            ctx.generated_text += clean
            ctx.printable_text += clean
            chunks, ctx.tools, ctx.printable_text = self._process_printable_text(
                ctx.printable_text, language_code, ctx.tools, ctx, runtime_config, response,
            )
            yield from chunks

        if ctx.sentence_batch:
            if ctx.printable_text.strip():
                ctx.sentence_batch.append(ctx.printable_text.strip())
                ctx.printable_text = ""
            yield LLMResponseChunk(text=" ".join(ctx.sentence_batch), language_code=language_code, runtime_config=runtime_config, response=response)
            ctx.sentence_batch = []

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def process(self, request: Transcription | GenerateResponseRequest):
        language_code = None
        runtime_config = None
        response = None
        ctx = StreamContext()

        if isinstance(request, GenerateResponseRequest):
            req = request
            runtime_config = req.runtime_config
            response = req.response
            original_chat = runtime_config.chat
            active_chat = original_chat.copy()
            language_code = req.language_code
            instructions = response.instructions if response and response.instructions else runtime_config.session.instructions
            tools = response.tools if response and response.tools else runtime_config.session.tools
            tool_choice = response.tool_choice if response and response.tool_choice else runtime_config.session.tool_choice
            self._apply_instructions(active_chat, instructions, tools, tool_choice, ctx)
            language_code, lang_name = resolve_auto_language(language_code)
            if lang_name:
                active_chat.append({"role": self.user_role, "content": f"Please reply to my message in {lang_name}."})
        elif isinstance(request, Transcription):
            original_chat = self.chat
            active_chat = original_chat
            logger.debug("infering language model...")
            language_code = request.language_code
            prompt_text = request.text
            language_code, lang_name = resolve_auto_language(language_code)
            if lang_name:
                prompt_text = f"Please reply to my message in {lang_name}. " + prompt_text
            active_chat.append({"role": self.user_role, "content": prompt_text})
        else:
            raise TypeError(f"Unexpected request type: {type(request)}")

        gen = self.cancel_scope.generation if self.cancel_scope else None

        yield from self._generate(active_chat, language_code, gen, ctx, runtime_config, response)

        if ctx.stopped:
            return

        original_chat.append({"role": "assistant", "content": ctx.generated_text})
        if ctx.tools:
            original_chat.append({"role": "assistant", "tool_calls": [
                {
                    "type": "function",
                    "id": t.call_id or "",
                    "function": {
                        "name": t.name or "",
                        "arguments": json.loads(t.arguments) if isinstance(t.arguments, str) else (t.arguments or {}),
                    },
                }
                for t in ctx.tools
            ]})
        original_chat.strip_images()
        logger.debug("Clean text: %s", ctx.generated_text)
        logger.info(f"Tools: {ctx.tools}")

        if not ctx.cancelled and (ctx.printable_text.strip() or ctx.tools):
            yield LLMResponseChunk(text=ctx.printable_text.strip(), language_code=language_code, tools=[t.model_dump() for t in ctx.tools], runtime_config=runtime_config, response=response)

        output_tokens = len(self.tokenizer.encode(ctx.raw_generated_text)) if ctx.raw_generated_text else 0
        if ctx.input_tokens or output_tokens:
            yield TokenUsage(input_tokens=ctx.input_tokens, output_tokens=output_tokens)
        yield EndOfResponse()

    def on_session_end(self) -> None:
        logger.debug("Language model session state reset")


# ======================================================================
# Text-only LLM
# ======================================================================


class LanguageModelHandler(BaseLanguageModelHandler):
    """Text-only language model handler (transformers or MLX backend)."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    gen_kwargs: dict
    torch_dtype: torch.dtype
    pipe: pipeline
    streamer: TextIteratorStreamer

    def setup(self, **kwargs: Any) -> None:
        super().setup(**kwargs)
        logger.info(f"LLM Backend: {self.backend}")
        self.warmup()

    def _load_model(self, model_name: str, device: str, torch_dtype: str, gen_kwargs: dict) -> None:

        logger.info("LLM Language Model Handler setup")

        if self.backend == "mlx":
            if not HAS_MLX:
                raise ImportError(
                    "mlx_lm is required for the 'mlx' backend. "
                    "Install with: pip install mlx-lm"
                )
            self.model, self.tokenizer = mlx_load(model_name)  # type: ignore[misc]
            self.gen_kwargs = gen_kwargs
        else:
            self.torch_dtype = getattr(torch, torch_dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)
            self.pipe = pipeline(
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

    def _generate(self, chat: Chat, language_code: str | None, gen: int | None, ctx: StreamContext, runtime_config=None, response=None) -> Iterator[LLMResponseChunk]:
        chat_messages = self._prepare_chat_for_transformers(chat.to_list())
        chat_input = self.tokenizer.apply_chat_template(chat_messages, tokenize=True)
        ctx.input_tokens += len(chat_input if isinstance(chat_input, list) else chat_input["input_ids"])
        logger.debug("Prompt token count: %d", ctx.input_tokens)

        chat_prompt = self.tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        if self.backend == "mlx":
            with MLXLockContext(handler_name="MLX-LLM", timeout=10.0):
                token_iter = mlx_stream_generate(
                    self.model,
                    self.tokenizer,
                    chat_prompt,
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
            thread = Thread(
                target=self.pipe, args=(chat_prompt,), kwargs=self.gen_kwargs
            )
            thread.start()
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx, runtime_config, response)
            self._finish_transformers_generation(thread)
            if self.device == "mps":
                torch.mps.empty_cache()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        dummy_input_text = "Repeat the word 'home'."
        dummy_chat = [{"role": self.user_role, "content": dummy_input_text}]
        n_steps = 2

        if self.backend == "mlx":
            for _ in range(n_steps):
                prompt = self.tokenizer.apply_chat_template(
                    dummy_chat, tokenize=False
                )
                mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
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
                thread = Thread(
                    target=self.pipe, args=(dummy_chat,), kwargs=warmup_gen_kwargs
                )
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

    model: AutoModelForImageTextToText
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    gen_kwargs: dict
    torch_dtype: torch.dtype
    streamer: TextIteratorStreamer

    def setup(self, **kwargs: Any) -> None:
        super().setup(**kwargs)
        logger.info(f"VLM Backend: {self.backend}")

    def _load_model(self, model_name: str, device: str, torch_dtype: str, gen_kwargs: dict) -> None:

        logger.info("VLM Language Model Handler setup")

        if self.backend == "mlx":
            if not HAS_MLX_VLM:
                raise ImportError(
                    "mlx-vlm is required for MLX VLM models. "
                    "Install with: pip install mlx-vlm"
                )
            self.model, self.processor = mlx_vlm_load(model_name)
            self.tokenizer = self.processor.tokenizer
            self.gen_kwargs = gen_kwargs
        else:
            self.torch_dtype = getattr(torch, torch_dtype)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=self.torch_dtype, trust_remote_code=True
            ).to(device)
            self.streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=1.0,
            )
            self._cancel_criteria = _CancelCriteria()
            self.gen_kwargs = gen_kwargs

    def _generate(self, chat: Chat, language_code: str | None, gen: int | None, ctx: StreamContext, runtime_config=None, response=None) -> Iterator[LLMResponseChunk]:
        prepared = self._prepare_chat_for_transformers(chat.to_list())
        if self.backend == "mlx":
            images, formatted_prompt = self._prepare_mlx_vlm_inputs(prepared)
            ctx.input_tokens += len(self.tokenizer.encode(formatted_prompt))
            logger.debug("MLX VLM prompt token count: %d", ctx.input_tokens)

            with MLXLockContext(handler_name="MLX-VLM", timeout=10.0):
                token_iter = mlx_vlm_stream_generate(
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
            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx, runtime_config, response)
            self._finish_transformers_generation(thread)
            if self.device == "mps":
                torch.mps.empty_cache()

    def _prepare_vlm_inputs(self, chat_messages: list[dict]) -> tuple[Any, int]:
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
            converted_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        return inputs, len(inputs["input_ids"][0])

    def _prepare_mlx_vlm_inputs(self, chat_messages: list[dict]) -> tuple[list, str]:
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
            converted_messages, tokenize=False, add_generation_prompt=True,
        )
        return images, formatted_prompt
