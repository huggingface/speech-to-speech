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
from pydantic import BaseModel, Field

from LLM.chat import Chat
from LLM.tool_call.function_call import extract_function_calls_from_text
from LLM.tool_call.function_tool import FunctionTool
from LLM.tool_call.tool_prompt import build_tool_system_prompt, build_block_regex, ENTER_CODE, END_CODE
from typing import Any, Literal
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from rich.console import Console
from LLM.utils import remove_unspeechable, image_url_to_pil
from LLM.voice_prompt import VOICE_SYSTEM_PROMPT
from api.openai_realtime.runtime_config import RuntimeConfig

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

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}


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
    cancelled: bool = False
    stopped: bool = False
    raw_generated_text: str = ""
    generated_text: str = ""
    printable_text: str = ""
    tools: list[dict] = Field(default_factory=list)
    input_tokens: int = 0

    @property
    def interrupted(self) -> bool:
        """True when generation ended early for any reason."""
        return self.cancelled or self.stopped


class BaseLanguageModelHandler(BaseHandler, ABC):
    """Abstract base for text-only and vision language model handlers.

    Holds shared pipeline logic (streaming, tool extraction, chat management)
    and delegates model-specific behaviour to three abstract hooks:
    ``_load_model``, ``_add_to_context``, and ``_generate``.
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

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            full_prompt = f"{VOICE_SYSTEM_PROMPT}\n\n{init_chat_prompt}"
            self.chat.init_chat({"role": init_chat_role, "content": full_prompt})
        self.user_role = user_role
        self.runtime_config = runtime_config
        self._last_instructions = init_chat_prompt
        self.tools = None
        self.tool_choice = None
        self._function_tools: list[FunctionTool] = []
        self._block_regex: str | None = None
        self._enter_code: str | None = None
        self._end_code: str | None = None

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
    def _add_to_context(self, role: str, content: str | list[dict]) -> None:
        """Handle an ``__ADD_TO_CONTEXT__`` payload (may contain images)."""

    @abstractmethod
    def _generate(
        self,
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

    def _apply_runtime_instructions(self) -> None:
        if not self.runtime_config:
            return
        new_instructions = self.runtime_config.session.instructions
        if not new_instructions:
            return

        raw_tools = self.runtime_config.session.tools or []
        self.tool_choice = self.runtime_config.session.tool_choice

        function_tools = [
            FunctionTool(**t.model_dump())
            for t in raw_tools
            if t.type == "function"
        ]
        tool_names = tuple(t.name for t in function_tools)
        old_tool_names = tuple(t.name for t in self._function_tools)

        instructions_changed = new_instructions != self._last_instructions
        tools_changed = tool_names != old_tool_names

        if not instructions_changed and not tools_changed:
            return

        self._last_instructions = new_instructions
        self._function_tools = function_tools
        self.tools = raw_tools

        if function_tools and self.tool_choice != "none":
            tool_section = build_tool_system_prompt(function_tools)
            full_instructions = f"{VOICE_SYSTEM_PROMPT}\n\n{new_instructions}\n\n{tool_section}"
            self._block_regex = build_block_regex()
            self._enter_code = ENTER_CODE
            self._end_code = END_CODE
        else:
            full_instructions = f"{VOICE_SYSTEM_PROMPT}\n\n{new_instructions}"
            self._block_regex = None
            self._enter_code = None
            self._end_code = None

        self.chat.init_chat({"role": "system", "content": full_instructions})
        logger.info(f"LLM instructions updated ({len(full_instructions)} chars, {len(function_tools)} tools)")

    def _extract_tools(self, text: str) -> tuple[str, list[dict]]:
        """Strip code blocks from *text* and return (clean_text, tool_dicts).

        When no tools are configured (``_block_regex is None``), returns
        the text unchanged with an empty tool list.
        """
        if not self._block_regex:
            return text, []
        clean_text, func_calls = extract_function_calls_from_text(
            text, self._block_regex,
        )
        tools = []
        for fc in func_calls:
            try:
                tools.append(
                    fc.to_realtime_function_tool_call(self._function_tools).model_dump()
                )
            except ValueError as e:
                logger.warning("Skipping invalid tool call: %s", e)
        return clean_text, tools

    def _process_printable_text(
        self, printable_text: str, language_code: str | None, tools: list[dict],
    ) -> tuple[list[tuple], list[dict], str]:
        """Extract complete code blocks and return complete sentences to yield.

        Returns ``(chunks_to_yield, updated_tools, remaining_printable_text)``.
        Each element in *chunks_to_yield* is a ``(text, language_code, [])`` tuple
        ready to be yielded to the downstream pipeline.
        """
        chunks: list[tuple] = []

        if self._block_regex and self._end_code and self._end_code in printable_text:
            stripped, func_calls = extract_function_calls_from_text(
                printable_text, self._block_regex,
            )
            for fc in func_calls:
                try:
                    tools.append(
                        fc.to_realtime_function_tool_call(self._function_tools).model_dump()
                    )
                except ValueError as e:
                    logger.warning("Skipping invalid tool call: %s", e)
            printable_text = stripped

        if self._enter_code and self._enter_code in printable_text:
            idx = printable_text.index(self._enter_code)
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
                ctx.printable_text, language_code, ctx.tools,
            )
            yield from chunks

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------

    def process(self, prompt: str | tuple) -> Iterator[tuple]:
        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__ADD_TO_CONTEXT__":
            _, role, content = prompt
            self._add_to_context(role, content)
            return

        if isinstance(prompt, tuple) and len(prompt) == 2 and prompt[0] == "__FUNCTION_RESULT__":
            _, result_text = prompt
            self.chat.append({"role": self.user_role, "content": result_text})
            return

        language_code = None

        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__GENERATE_RESPONSE__":
            _, override_instructions, _ = prompt
            self._apply_runtime_instructions()
            if override_instructions:
                self.chat.append({"role": self.user_role, "content": override_instructions})
        else:
            self._apply_runtime_instructions()
            logger.debug("infering language model...")
            if isinstance(prompt, tuple):
                prompt, language_code = prompt
                if language_code[-5:] == "-auto":
                    language_code = language_code[:-5]
                    prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt
            self.chat.append({"role": self.user_role, "content": prompt})

        gen = self.cancel_scope.generation if self.cancel_scope else None
        ctx = StreamContext()

        yield from self._generate(language_code, gen, ctx)

        if ctx.stopped:
            return

        self.chat.append({"role": "assistant", "content": ctx.generated_text})
        self.chat.strip_images()
        logger.debug("Clean text: %s", ctx.generated_text)
        logger.info(f"Tools: {ctx.tools}")

        if not ctx.cancelled and (ctx.printable_text.strip() or ctx.tools):
            yield (ctx.printable_text.strip(), language_code, ctx.tools)

        output_tokens = len(self.tokenizer.encode(ctx.raw_generated_text)) if ctx.raw_generated_text else 0
        if ctx.input_tokens or output_tokens:
            yield ("__TOKEN_USAGE__", ctx.input_tokens, output_tokens)
        yield ("__END_OF_RESPONSE__", None, None)

    def on_session_end(self) -> None:
        # reset() also clears init_chat_message, so a previous session's
        # instructions cannot persist into the next one.
        self.chat.reset()
        self._last_instructions = None
        self.tools = None
        self.tool_choice = None
        self._function_tools = []
        self._block_regex = None
        self._enter_code = None
        self._end_code = None
        logger.debug("Language model session state reset (chat + tool cache)")


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

    def _add_to_context(self, role: str, content: str | list[dict]) -> None:
        if isinstance(content, list):
            for p in content:
                if p.get("type") == "input_text" and p.get("text"):
                    self.chat.append({"role": role, "content": p["text"]})
                elif p.get("type") == "input_image":
                    logger.debug("Dropped image content (LLM text-only)")
        else:
            self.chat.append({"role": role, "content": content})

    def _generate(self, language_code: str | None, gen: int | None, ctx: StreamContext) -> Iterator[tuple[str, str | None, list]]:
        chat_input = self.tokenizer.apply_chat_template(self.chat.to_list(), tokenize=True)
        ctx.input_tokens = len(chat_input if isinstance(chat_input, list) else chat_input["input_ids"])
        logger.debug("Prompt token count: %d", ctx.input_tokens)

        chat_prompt = self.tokenizer.apply_chat_template(
            self.chat.to_list(), tokenize=False, add_generation_prompt=True, enable_thinking=False
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

    def _add_to_context(self, role: str, content: str | list[dict]) -> None:
        self.chat.append({"role": role, "content": content})

    def _generate(self, language_code: str | None, gen: int | None, ctx: StreamContext) -> Iterator[tuple[str, str | None, list]]:
        if self.backend == "mlx":
            images, formatted_prompt = self._prepare_mlx_vlm_inputs(self.chat.to_list())
            ctx.input_tokens = len(self.tokenizer.encode(formatted_prompt))
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
            inputs, ctx.input_tokens = self._prepare_vlm_inputs(self.chat.to_list())
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
