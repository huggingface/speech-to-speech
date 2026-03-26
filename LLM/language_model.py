from threading import Event, Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer,
)
import torch
import logging
from nltk import sent_tokenize

from LLM.chat import Chat
from LLM.tool_call.function_call import extract_function_calls_from_text
from LLM.tool_call.function_tool import FunctionTool
from LLM.tool_call.tool_prompt import build_tool_system_prompt, build_block_regex, ENTER_CODE, END_CODE
from typing import Literal
from baseHandler import BaseHandler
from rich.console import Console
from LLM.utils import remove_emojis
from api.openai_realtime.runtime_config import RuntimeConfig

try:
    from mlx_lm import load as mlx_load, stream_generate as mlx_stream_generate, generate as mlx_generate
    from utils.mlx_lock import MLXLockContext
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

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


class LanguageModelHandler(BaseHandler):
    """
    Handles the language model part.
    """

    def setup(
        self,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        device="cuda",
        torch_dtype="float16",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
        runtime_config: RuntimeConfig | None = None,
        cancel_response: Event | None = None,
        backend: Literal["transformers", "mlx"] = "transformers",
    ):
        self.backend = backend
        self.cancel_response = cancel_response
        self.device = device
        self.model_name = model_name

        logger.info(f"LLM Backend: {self.backend}")

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
            )
            self.gen_kwargs = {
                "streamer": self.streamer,
                "return_full_text": False,
                **gen_kwargs,
            }

        self.chat = Chat(chat_size)
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial promt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        self.user_role = user_role
        self.runtime_config = runtime_config
        self._last_instructions = init_chat_prompt
        self.tools = None
        self.tool_choice = None
        self._function_tools: list[FunctionTool] = []
        self._block_regex: str | None = None
        self._enter_code: str | None = None
        self._end_code: str | None = None

        self.warmup()

    def warmup(self):
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

    def _apply_runtime_instructions(self):
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
            full_instructions = f"{new_instructions}\n\n{tool_section}"
            self._block_regex = build_block_regex()
            self._enter_code = ENTER_CODE
            self._end_code = END_CODE
        else:
            full_instructions = new_instructions
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
        self, printable_text: str, language_code, tools: list[dict],
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

    def process(self, prompt):
        if isinstance(prompt, tuple) and len(prompt) == 3 and prompt[0] == "__ADD_TO_CONTEXT__":
            _, role, text = prompt
            self.chat.append({"role": role, "content": text})
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

        if logger.isEnabledFor(logging.DEBUG):
            chat_input = self.tokenizer.apply_chat_template(self.chat.to_list(), tokenize=True)
            num_tokens = len(chat_input["input_ids"])
            logger.info("Prompt token count: %d", num_tokens)

        chat_prompt = self.tokenizer.apply_chat_template(
            self.chat.to_list(), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        cancelled = False

        # TODO: Rethink stream generation to use special yield tags that signal
        # the engine whether the model is sending a partial or complete
        # LLM response. Enable TTS response only on complete LLM response (and send partial events for realtime engine).
        if self.backend == "mlx":
            generated_text = ""
            printable_text = ""
            tools: list[dict] = []
            with MLXLockContext(handler_name="MLX-LLM", timeout=10.0):
                for t in mlx_stream_generate(
                    self.model,
                    self.tokenizer,
                    chat_prompt,
                    max_tokens=self.gen_kwargs["max_new_tokens"],
                ):
                    if self.cancel_response and self.cancel_response.is_set():
                        logger.info("LLM generation cancelled (interruption)")
                        cancelled = True
                        break
                    new_text = remove_emojis(t.text)
                    generated_text += new_text
                    printable_text += new_text
                    chunks, tools, printable_text = self._process_printable_text(
                        printable_text, language_code, tools,
                    )
                    for chunk in chunks:
                        yield chunk
            try:
                mx.clear_cache()
            except Exception:
                pass
            torch.mps.empty_cache()
        else:
            thread = Thread(
                target=self.pipe, args=(chat_prompt,), kwargs=self.gen_kwargs
            )
            thread.start()
            generated_text = ""
            printable_text = ""
            tools: list[dict] = []
            for new_text in self.streamer:
                if self.cancel_response and self.cancel_response.is_set():
                    logger.info("LLM generation cancelled (interruption)")
                    cancelled = True
                    break
                new_text = remove_emojis(new_text)
                generated_text += new_text
                printable_text += new_text
                chunks, tools, printable_text = self._process_printable_text(
                    printable_text, language_code, tools,
                )
                for chunk in chunks:
                    yield chunk
            if self.device == "mps":
                torch.mps.empty_cache()

        self.chat.append({"role": "assistant", "content": generated_text})
        logger.debug("chat: %s", self.chat.to_list())
        logger.debug("generated_text: %s", generated_text)

        if not cancelled and (printable_text.strip() or tools):
            yield (printable_text.strip(), language_code, tools)
        yield ("__END_OF_RESPONSE__", None, None)

    def on_session_end(self):
        self.chat.reset()
        logger.debug("Language model chat state reset")
