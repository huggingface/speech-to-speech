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
import torch
import logging
from queue import Empty
from nltk import sent_tokenize
from pydantic import BaseModel, ConfigDict, Field

from LLM.chat import Chat
from LLM.tool_call.function_call import extract_function_calls_from_text
from LLM.tool_call.function_tool import FunctionTool
from LLM.tool_call.tool_prompt import build_tool_system_prompt, build_block_regex, ENTER_CODE, END_CODE
from typing import Any, Literal
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from rich.console import Console
from LLM.utils import remove_unspeechable, resolve_auto_language, image_url_to_pil
from LLM.voice_prompt import build_voice_system_prompt
from api.openai_realtime.runtime_config import RuntimeConfig
from pipeline_messages import GenerateRequest, MessageTag

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

console = Console()


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
    tools: list[dict] = Field(default_factory=list)
    function_tools: list[FunctionTool] = Field(default_factory=list)
    block_regex: str | None = None
    enter_code: str | None = None
    end_code: str | None = None
    input_tokens: int = 0

    @property
    def interrupted(self) -> bool:
        """True when generation ended early for any reason."""
        return self.cancelled or self.stopped


class BaseLanguageModelHandler(BaseHandler, ABC):
    """Abstract base for text-only and vision language model handlers.

    Holds shared pipeline logic (streaming, tool extraction, chat management)
    and delegates model-specific behaviour to three abstract hooks:
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
        runtime_config: RuntimeConfig | None = None,
        cancel_scope: CancelScope | None = None,
        backend: Literal["transformers", "mlx"] = "transformers",
        enable_thinking: bool = False,
    ):
        self.backend = backend
        self.cancel_scope = cancel_scope
        self.device = device
        self.model_name = model_name
        self.enable_thinking = enable_thinking

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
        self.runtime_config = runtime_config

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
    ) -> Iterator[tuple[str, str | None, list]]:
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

    def _apply_runtime_instructions(self, ctx: StreamContext) -> None:
        """Legacy path: read instructions/tools from runtime_config (non-realtime)."""
        if not self.runtime_config:
            return
        new_instructions = self.runtime_config.session.instructions
        if not new_instructions:
            return
        raw_tools = self.runtime_config.session.tools or []
        tool_choice = self.runtime_config.session.tool_choice
        self._apply_instructions(self.chat, new_instructions, raw_tools, tool_choice, ctx)

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

    def _extract_tools(self, text: str, ctx: StreamContext) -> tuple[str, list[dict]]:
        """Strip code blocks from *text* and return (clean_text, tool_dicts).

        When no tools are configured (``ctx.block_regex is None``), returns
        the text unchanged with an empty tool list.
        """
        if not ctx.block_regex:
            return text, []
        clean_text, func_calls = extract_function_calls_from_text(
            text, ctx.block_regex,
        )
        tools = []
        for fc in func_calls:
            try:
                tools.append(
                    fc.to_realtime_function_tool_call(ctx.function_tools).model_dump()
                )
            except ValueError as e:
                logger.warning("Skipping invalid tool call: %s", e)
        return clean_text, tools

    def _process_printable_text(
        self, printable_text: str, language_code: str | None, tools: list[dict],
        ctx: StreamContext,
    ) -> tuple[list[tuple], list[dict], str]:
        """Extract complete code blocks and return complete sentences to yield.

        Returns ``(chunks_to_yield, updated_tools, remaining_printable_text)``.
        Each element in *chunks_to_yield* is a ``(text, language_code, [])`` tuple
        ready to be yielded to the downstream pipeline.
        """
        chunks: list[tuple] = []

        if ctx.block_regex and ctx.end_code and ctx.end_code in printable_text:
            stripped, func_calls = extract_function_calls_from_text(
                printable_text, ctx.block_regex,
            )
            for fc in func_calls:
                try:
                    tools.append(
                        fc.to_realtime_function_tool_call(ctx.function_tools).model_dump()
                    )
                except ValueError as e:
                    logger.warning("Skipping invalid tool call: %s", e)
            printable_text = stripped

        if ctx.enter_code and ctx.enter_code in printable_text:
            idx = printable_text.index(ctx.enter_code)
            before = printable_text[:idx]
            if before.strip():
                for s in sent_tokenize(before):
                    chunks.append((s, language_code, []))
            printable_text = printable_text[idx:]
            return chunks, tools, printable_text

        if printable_text:
            sentences = sent_tokenize(printable_text)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    chunks.append((s, language_code, []))
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
    ) -> Iterator[tuple[str, str | None, list]]:
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
                ctx.printable_text, language_code, ctx.tools, ctx,
            )
            yield from chunks

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def process(self, prompt: str | tuple) -> Iterator[tuple]:
        language_code = None
        ctx = StreamContext()

        if isinstance(prompt, tuple) and len(prompt) == 2 and prompt[0] == MessageTag.GENERATE_RESPONSE:
            req: GenerateRequest = prompt[1]
            original_chat = req.chat
            active_chat = original_chat.copy()
            language_code = req.language_code
            self._apply_instructions(active_chat, req.instructions, req.tools, req.tool_choice, ctx)
            if req.override_instructions:
                active_chat.append({"role": self.user_role, "content": req.override_instructions})
            language_code, lang_name = resolve_auto_language(language_code)
            if lang_name:
                active_chat.append({"role": self.user_role, "content": f"Please reply to my message in {lang_name}."})
        else:
            original_chat = self.chat
            active_chat = original_chat
            self._apply_runtime_instructions(ctx)
            logger.debug("infering language model...")
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                language_code, lang_name = resolve_auto_language(language_code)
                if lang_name:
                    prompt = f"Please reply to my message in {lang_name}. " + prompt
            active_chat.append({"role": self.user_role, "content": prompt})

        gen = self.cancel_scope.generation if self.cancel_scope else None

        yield from self._generate(active_chat, language_code, gen, ctx)

        if ctx.stopped:
            return

        original_chat.append({"role": "assistant", "content": ctx.generated_text})
        original_chat.strip_images()
        logger.debug("Clean text: %s", ctx.generated_text)
        logger.info(f"Tools: {ctx.tools}")

        if not ctx.cancelled and (ctx.printable_text.strip() or ctx.tools):
            yield (ctx.printable_text.strip(), language_code, ctx.tools)

        output_tokens = len(self.tokenizer.encode(ctx.raw_generated_text)) if ctx.raw_generated_text else 0
        if ctx.input_tokens or output_tokens:
            yield (MessageTag.TOKEN_USAGE, ctx.input_tokens, output_tokens)
        yield (MessageTag.END_OF_RESPONSE, None, None)

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

    def _generate(self, chat: Chat, language_code: str | None, gen: int | None, ctx: StreamContext) -> Iterator[tuple[str, str | None, list]]:
        chat_messages = chat.to_list()
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
                yield from self._stream_tokens(token_iter, gen, language_code, ctx)
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
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx)
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
            print(self.processor)
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

    def _generate(self, chat: Chat, language_code: str | None, gen: int | None, ctx: StreamContext) -> Iterator[tuple[str, str | None, list]]:
        if self.backend == "mlx":
            images, formatted_prompt = self._prepare_mlx_vlm_inputs(chat.to_list())
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
                yield from self._stream_tokens(token_iter, gen, language_code, ctx)
                self._finish_mlx_generation(token_iter)
            try:
                mx.clear_cache()
            except Exception:
                pass
            torch.mps.empty_cache()
        else:
            inputs, input_tokens = self._prepare_vlm_inputs(chat.to_list())
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
            yield from self._stream_tokens(self.streamer, gen, language_code, ctx)
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
