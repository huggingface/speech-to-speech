#!/usr/bin/env python3
"""
Function parser for extracting function names, parameter names, and values from string function calls.

Uses Python's ``tokenize`` and ``ast`` modules so that nested parentheses,
strings containing ')' characters, tuples, dicts, etc. are handled correctly.
"""

import ast
import io
import json
import logging
import re
import tokenize
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel

from LLM.tool_call.function_tool import FunctionTool

logger = logging.getLogger(__name__)

_POSITIONAL_RE = re.compile(r"^__arg_\d+__$")


# ── AST / tokenize helpers ───────────────────────────────────────────


def _split_top_level_calls(source: str) -> List[str]:
    """Split *source* into individual ``name(...)`` expression strings.

    Uses the tokenizer to walk tokens and track parenthesis depth so that
    nested parens, strings with ')' chars, etc. are handled correctly.
    """
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    calls: List[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]
        if tok.type != tokenize.NAME:
            i += 1
            continue

        start = i
        j = i + 1

        # Walk past dotted attribute access (e.g. ``mobile.click``)
        while (
            j + 1 < len(tokens)
            and tokens[j].string == "."
            and tokens[j + 1].type == tokenize.NAME
        ):
            j += 2

        if j >= len(tokens) or tokens[j].string != "(":
            i += 1
            continue

        # Track balanced parens
        depth = 0
        end = None
        k = j
        while k < len(tokens):
            t = tokens[k]
            if t.type == tokenize.OP and t.string == "(":
                depth += 1
            elif t.type == tokenize.OP and t.string == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
            k += 1

        if end is None:
            i += 1
            continue

        calls.append(tokenize.untokenize(tokens[start : end + 1]).strip())
        i = end + 1

    return calls


def _extract_function_name(node: ast.expr) -> str:
    """Return the dotted function name from a Call node's ``func`` attribute."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _extract_function_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    raise ValueError(f"Unsupported function target: {ast.dump(node)}")


def _literal_from_ast(node: ast.AST) -> Any:
    """Convert an AST node to a Python literal value."""
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.List):
        return [_literal_from_ast(elt) for elt in node.elts]

    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(elt) for elt in node.elts]

    if isinstance(node, ast.Dict):
        return {
            _literal_from_ast(key): _literal_from_ast(value)
            for key, value in zip(node.keys, node.values)
        }

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_from_ast(node.operand)
        if not isinstance(value, (int, float)):
            raise ValueError(f"Unsupported unary literal: {ast.dump(node)}")
        return -value if isinstance(node.op, ast.USub) else value

    raise ValueError(f"Unsupported literal: {ast.dump(node)}")


def _parse_call_expr(expr: str) -> "FunctionToolCall":
    """Parse a single ``name(args...)`` expression string into a FunctionToolCall."""
    parsed = ast.parse(expr, mode="eval").body
    if not isinstance(parsed, ast.Call):
        raise ValueError(f"Expression is not a function call: {expr!r}")

    parameters: "OrderedDict[str, Any]" = OrderedDict()

    for idx, arg in enumerate(parsed.args):
        parameters[f"__arg_{idx}__"] = _literal_from_ast(arg)

    for kw in parsed.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs are not supported")
        parameters[kw.arg] = _literal_from_ast(kw.value)

    return FunctionToolCall(
        function_name=_extract_function_name(parsed.func),
        parameters=parameters,
        original_string=expr,
    )


# ── Data model ───────────────────────────────────────────────────────


class FunctionToolCall(BaseModel):
    """Represents a parsed function call with its parameters."""

    function_name: str
    parameters: Dict[str, Any]
    original_string: str
    description: str = ""

    def to_realtime_function_tool_call(
        self,
        function_tools: list[FunctionTool] | None = None,
    ) -> ResponseFunctionToolCall:
        positional = {k for k in self.parameters if _POSITIONAL_RE.match(k)}
        if positional:
            logger.warning(
                "Dropping positional arguments for '%s': %s",
                self.function_name,
                positional,
            )
        arguments = {
            k: v for k, v in self.parameters.items() if not _POSITIONAL_RE.match(k)
        }

        if function_tools is not None:
            tool = next(
                (t for t in function_tools if t.name == self.function_name),
                None,
            )
            if tool is None:
                available = [t.name for t in function_tools]
                raise ValueError(
                    f"Function '{self.function_name}' not found in available tools: {available}"
                )

            schema = tool.parameters if isinstance(tool.parameters, dict) else {}
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

            undeclared = {k for k in arguments if k not in properties}
            if undeclared:
                logger.warning(
                    "Dropping undeclared parameters for '%s': %s",
                    self.function_name,
                    undeclared,
                )
                arguments = {k: v for k, v in arguments.items() if k in properties}

            missing = required - set(arguments.keys())
            if missing:
                raise ValueError(
                    f"Missing required parameters for '{self.function_name}': {missing}"
                )

        return ResponseFunctionToolCall(
            name=self.function_name,
            arguments=json.dumps(arguments),
            call_id=f"call_{uuid4().hex[:12]}",
            type="function_call",
        )


# ── Public API ───────────────────────────────────────────────────────


def parse_function_call(
    function_string: str, pattern_to_match: list[str] = []
) -> List[FunctionToolCall]:
    """Parse a function call string and extract all function calls found.

    Args:
        function_string: String representation of function calls.
        pattern_to_match: If non-empty, only calls whose function name
            contains at least one of these substrings are returned.

    Returns:
        List of FunctionToolCall objects with parsed information.
    """
    function_string = function_string.strip()
    if not function_string:
        return []

    results: List[FunctionToolCall] = []
    for expr in _split_top_level_calls(function_string):
        call = _parse_call_expr(expr)
        if pattern_to_match and all(
            pattern not in call.function_name for pattern in pattern_to_match
        ):
            continue
        results.append(call)
    return results


def parse_multiple_functions(function_strings: List[str]) -> List[FunctionToolCall]:
    """Parse multiple function call strings.

    Args:
        function_strings: List of function call strings.

    Returns:
        List of FunctionToolCall objects.
    """
    results: List[FunctionToolCall] = []
    for func_str in function_strings:
        try:
            results.extend(parse_function_call(func_str))
        except Exception:
            continue
    return results


def extract_function_calls_from_text(
    text: str, block_regex: str = ".*"
) -> Tuple[str, List[FunctionToolCall]]:
    """Extract function calls from delimited code blocks inside *text*.

    The LLM is prompted to wrap tool calls inside code-block delimiters
    (e.g. ``<code>func(x=1)</code>``).  This function finds those blocks,
    parses the function calls within them, and returns the remaining text
    (with blocks stripped) alongside the parsed calls.

    Args:
        text: Full model output potentially containing code blocks.
        block_regex: Regex matching the code-block delimiters **and** their
            content (e.g. ``r"<code>.*?</code>"``).  Only text **inside**
            matched blocks is scanned for function calls.

    Returns:
        ``(outside_text, function_calls)`` -- the text with blocks stripped
        and the parsed function calls found inside the blocks.
    """
    if not block_regex:
        return text, []

    matches = list(re.finditer(block_regex, text, flags=re.DOTALL))
    if not matches:
        return text, []

    outside = re.sub(block_regex, "", text, flags=re.DOTALL)
    inside = " ".join(match.group(0) for match in matches).strip()
    if not inside:
        return outside, []

    try:
        return outside, parse_function_call(inside)
    except Exception:
        return outside, []
