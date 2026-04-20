"""
Standalone Qwen3 Coder XML tool-call parser inspire by official Qwen3-Coder Tool Parser:


https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/qwen3coder_tool_parser_vllm.py


Extracts ``<tool_call><function=name><parameter=key>value</parameter></function></tool_call>``
blocks from model output and returns ``list[ResponseFunctionToolCall]``.

No vLLM dependency -- only the ``openai`` SDK (already in the project) and the
standard library.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import uuid
from typing import Any

from nltk import sent_tokenize
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.function_tool import FunctionTool

logger = logging.getLogger(__name__)

_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL,
)
_FUNCTION_REGEX = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
_PARAMETER_REGEX = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
_FUNCTION_PREFIX = "<function="

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# Speech-only segments for chat history: split on complete blocks or unclosed ``<tool_call>…`` tail.
_TOOL_CALL_SPEECH_SPLIT = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_call>.*$", re.DOTALL,
)


def strip_qwen_tool_markup_for_chat(text: str) -> str:
    """Remove tool XML and return speech-only text (assistant history and final TTS buffer).

    Splits on each ``<tool_call>…</tool_call>`` span and on an unclosed ``<tool_call>…``
    tail, then joins the non-empty segments (text before, between, and after blocks)
    with a single space.
    """
    parts = _TOOL_CALL_SPEECH_SPLIT.split(text)
    segments = [p.strip() for p in parts if p.strip()]
    return " ".join(segments)


def _strip_leading_orphan_closes(text: str) -> str:
    """Remove ``</tool_call>`` that appear before any ``<tool_call>`` (malformed / streaming glitches)."""
    while True:
        first_open = text.find(TOOL_CALL_OPEN)
        first_close = text.find(TOOL_CALL_CLOSE)
        if first_close != -1 and (first_open == -1 or first_close < first_open):
            text = (
                text[:first_close]
                + text[first_close + len(TOOL_CALL_CLOSE) :]
            )
            continue
        break
    return text


def _flush_complete_sentences(text: str, *, flush_single: bool) -> tuple[list[str], str]:
    """Split *text* on sentence boundaries for streaming TTS.

    When *flush_single* is True (speech before an unclosed ``<tool_call>``), a lone sentence
    is yielded and the remainder is empty. When False (normal tail), a lone sentence stays
    in the remainder buffer unchanged.
    """
    if not text.strip():
        return ([], "") if flush_single else ([], text)
    sentences = sent_tokenize(text)
    if len(sentences) > 1:
        return sentences[:-1], sentences[-1]
    if len(sentences) == 1:
        if flush_single:
            return [sentences[0]], ""
        return [], text
    return ([], "") if flush_single else ([], text)


def process_printable_text_qwen_xml(
    printable_text: str,
    tools: list[dict],
    parser: Qwen3CoderToolParser,
) -> tuple[list[str], list[dict], str]:
    """Extract closed tool blocks and speech sentence strings for streaming TTS.

    Returns ``(chunks_to_yield, updated_tools, remaining_printable_text)``.
    Each chunk is a single sentence string (no tool XML). Call ``parse()`` only on complete
    ``<tool_call>…</tool_call>`` spans. The caller attaches ``language_code`` and tool metadata.
    """
    chunks: list[str] = []

    printable_text = _strip_leading_orphan_closes(printable_text)

    while True:
        start = printable_text.find(TOOL_CALL_OPEN)
        if start == -1:
            break
        end = printable_text.find(TOOL_CALL_CLOSE, start)
        if end == -1:
            break
        end_close = end + len(TOOL_CALL_CLOSE)
        span = printable_text[start:end_close]
        for tc in parser.parse(span):
            tools.append(tc.model_dump())
        printable_text = printable_text[:start] + printable_text[end_close:]

    printable_text = _strip_leading_orphan_closes(printable_text)

    if TOOL_CALL_OPEN in printable_text:
        idx = printable_text.index(TOOL_CALL_OPEN)
        before = printable_text[:idx]
        tail_from_tag = printable_text[idx:]
        if before.strip():
            flushed, remainder = _flush_complete_sentences(before, flush_single=True)
            chunks.extend(flushed)
            printable_text = (
                (remainder + tail_from_tag) if remainder else tail_from_tag
            )
        else:
            printable_text = tail_from_tag
        return chunks, tools, printable_text

    if printable_text:
        flushed, remainder = _flush_complete_sentences(
            printable_text, flush_single=False,
        )
        chunks.extend(flushed)
        printable_text = remainder

    return chunks, tools, printable_text


class Qwen3CoderToolParser:
    """Parse Qwen3-style XML tool calls into ``ResponseFunctionToolCall`` objects.

    Parameters
    ----------
    tools:
        Optional list of ``FunctionTool`` definitions.  When provided the
        parser uses the JSON-Schema ``parameters`` to convert raw string
        values to the declared types (int, float, bool, object, ...).
        Without tool definitions every parameter value stays a string.
    """

    def __init__(self, tools: list[FunctionTool] | None = None) -> None:
        self.tools = tools

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, model_output: str) -> list[ResponseFunctionToolCall]:
        """Extract all tool calls from *model_output*.

        Returns an empty list when the output contains no tool calls.
        """
        if _FUNCTION_PREFIX not in model_output:
            return []

        try:
            function_call_strs = self._get_function_calls(model_output)
            if not function_call_strs:
                return []

            tool_calls: list[ResponseFunctionToolCall] = []
            for fc_str in function_call_strs:
                tc = self._parse_xml_function_call(fc_str)
                if tc is not None:
                    tool_calls.append(tc)
            return tool_calls

        except Exception:
            logger.exception("Error extracting tool calls from response.")
            return []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_call_id() -> str:
        return f"call_{uuid.uuid4().hex[:24]}"

    @staticmethod
    def _get_function_calls(model_output: str) -> list[str]:
        matched_ranges = _TOOL_CALL_REGEX.findall(model_output)
        raw_tool_calls = [m[0] or m[1] for m in matched_ranges]

        if not raw_tool_calls:
            raw_tool_calls = [model_output]

        raw_function_calls: list[str] = []
        for tc in raw_tool_calls:
            raw_function_calls.extend(_FUNCTION_REGEX.findall(tc))

        return [m[0] or m[1] for m in raw_function_calls]

    def _parse_xml_function_call(
        self, function_call_str: str,
    ) -> ResponseFunctionToolCall | None:
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name)
        body = function_call_str[end_index + 1:]

        param_dict: dict[str, Any] = {}
        for match_text in _PARAMETER_REGEX.findall(body):
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1:])

            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name,
            )

        return ResponseFunctionToolCall(
            name=function_name,
            arguments=json.dumps(param_dict, ensure_ascii=False),
            call_id=self._generate_call_id(),
            type="function_call",
        )

    def _get_arguments_config(self, func_name: str) -> dict:
        """Return the ``properties`` dict from the tool schema for *func_name*."""
        if self.tools is None:
            return {}
        for tool in self.tools:
            if getattr(tool, "type", None) != "function":
                continue
            if getattr(tool, "name", None) != func_name:
                continue
            params = getattr(tool, "parameters", None)
            if not isinstance(params, dict):
                return {}
            return params.get("properties", params)
        logger.warning("Tool '%s' is not defined in the tools list.", func_name)
        return {}

    @staticmethod
    def _convert_param_value(
        param_value: str,
        param_name: str,
        param_config: dict,
        func_name: str,
    ) -> Any:
        """Convert a raw string value to the type declared in the tool schema."""
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config:
                logger.warning(
                    "Parsed parameter '%s' is not defined in tool '%s' parameters, "
                    "returning as string.",
                    param_name, func_name,
                )
            return param_value

        spec = param_config[param_name]
        if isinstance(spec, dict) and "type" in spec:
            param_type = str(spec["type"]).strip().lower()
        else:
            param_type = "string"

        if param_type in ("string", "str", "text", "varchar", "char", "enum"):
            return param_value

        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.warning(
                    "Value '%s' of '%s' is not an integer in tool '%s', keeping string.",
                    param_value, param_name, func_name,
                )
                return param_value

        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                has_decimal = "." in param_value or "e" in param_value.lower()
                val = float(param_value)
                if not has_decimal and val.is_integer():
                    return int(val)
                return val
            except (ValueError, TypeError):
                logger.warning(
                    "Value '%s' of '%s' is not a float in tool '%s', keeping string.",
                    param_value, param_name, func_name,
                )
                return param_value

        if param_type in ("boolean", "bool", "binary"):
            lower = param_value.lower()
            if lower not in ("true", "false"):
                logger.warning(
                    "Value '%s' of '%s' is not a boolean in tool '%s', defaulting to false.",
                    param_value, param_name, func_name,
                )
            return lower == "true"

        if (
            param_type in ("object", "array", "arr")
            or param_type.startswith("dict")
            or param_type.startswith("list")
        ):
            try:
                return json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "Value '%s' of '%s' failed json.loads in tool '%s', trying literal_eval.",
                    param_value, param_name, func_name,
                )

        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            logger.warning(
                "Value '%s' of '%s' could not be converted in tool '%s', keeping string.",
                param_value, param_name, func_name,
            )
            return param_value
