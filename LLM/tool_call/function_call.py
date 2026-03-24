#!/usr/bin/env python3
"""
Function parser for extracting function names, parameter names, and values from string function calls.
Supports both mobile and pyautogui function patterns.
"""

import re
from collections import OrderedDict
import json
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel
from openai.types.responses import ResponseFunctionToolCall
from LLM.tool_call.function_tool import FunctionTool
from uuid import uuid4


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
        arguments = {
            name: value for name, value in self.parameters.items()
            if not name.startswith("arg_")
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

            # Drop arguments not declared in the tool schema
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


def parse_function_call(
    function_string: str, pattern_to_match: list[str] = []
) -> List[FunctionToolCall]:
    """
    Parse a function call string and extract all function calls found.

    Args:
        function_string: String representation of function calls

    Returns:
        List of FunctionToolCall objects with parsed information

    Examples:
        >>> parse_function_call("mobile.wait(seconds=3)")
        [FunctionToolCall(function_name='wait', parameters={'seconds': 3}, ...)]

        >>> parse_function_call("mobile. wait(seconds=3)")
        [FunctionToolCall(function_name='wait', parameters={'seconds': 3}, ...)]

        >>> parse_function_call("mobile.wait(seconds=3) mobile.home()")
        [FunctionToolCall(function_name='wait', parameters={'seconds': 3}, ...), FunctionToolCall(function_name='home', parameters={}, ...)]
    """
    # Remove any leading/trailing whitespace
    function_string = function_string.strip()

    # Pattern to match function calls with parameters
    # Matches: function_name(param1=value1, param2=value2, ...)
    # Can have any characters before the function call, extracts just the function name
    pattern = r".*?([a-zA-Z_][a-zA-Z0-9_.]*)\(([^)]*)\)"

    matches = re.findall(pattern, function_string)
    if not matches:
        # No valid function calls found in: {function_string}
        return []

    results = []
    for match in matches:
        function_name = match[0]
        params_string = match[1]

        if pattern_to_match and all(
            pattern not in function_name for pattern in pattern_to_match
        ):
            continue

        # Parse parameters
        parameters = parse_parameters(params_string)

        # Create the original string for this specific function call
        original_string = f"{function_name}({params_string})"

        results.append(
            FunctionToolCall(
                function_name=function_name,
                parameters=parameters,
                original_string=original_string,
            )
        )

    return results


def parse_parameters(params_string: str) -> Dict[str, Any]:
    """
    Parse parameter string and extract parameter names and values.

    Args:
        params_string: String containing parameters (e.g., "x=0.5, y=0.6, text='hello'")

    Returns:
        Dictionary mapping parameter names to their values

    Examples:
        >>> parse_parameters("x=0.5, y=0.6")
        {'x': 0.5, 'y': 0.6}

        >>> parse_parameters("app_name='drupe'")
        {'app_name': 'drupe'}

        >>> parse_parameters("'text'")
        {'arg_0': 'text'}

        >>> parse_parameters("1, 3, 4")
        {'arg_0': 1, 'arg_1': 3, 'arg_2': 4}

        >>> parse_parameters("arg1, arg2, x=0.5")
        {'arg_0': 'arg1', 'arg_1': 'arg2', 'x': 0.5}
    """
    if not params_string.strip():
        return {}

    parameters = OrderedDict()

    # Split by commas, but be careful with commas inside quotes or brackets
    param_parts = split_parameters(params_string)

    positional_index = 0

    for part in param_parts:
        part = part.strip()
        if not part:
            continue

        # Parse individual parameter
        name, value = parse_single_parameter(part)

        # For positional arguments, use index-based naming
        if name.startswith("arg_"):
            name = f"arg_{positional_index}"
            positional_index += 1

        parameters[name] = value

    return parameters


def split_parameters(params_string: str) -> List[str]:
    """
    Split parameter string by commas, respecting quotes and brackets.

    Args:
        params_string: String containing parameters

    Returns:
        List of individual parameter strings
    """
    parts = []
    current_part = ""
    paren_count = 0
    bracket_count = 0
    brace_count = 0
    in_quotes = False
    quote_char = None

    for char in params_string:
        if char in ['"', "'"] and (not in_quotes or char == quote_char):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            else:
                in_quotes = False
                quote_char = None
        elif not in_quotes:
            if char == "(":
                paren_count += 1
            elif char == ")":
                paren_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1
            elif char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif (
                char == ","
                and paren_count == 0
                and bracket_count == 0
                and brace_count == 0
            ):
                parts.append(current_part.strip())
                current_part = ""
                continue

        current_part += char

    if current_part.strip():
        parts.append(current_part.strip())

    return parts


def parse_single_parameter(param_string: str) -> Tuple[str, Any]:
    """
    Parse a single parameter string into name and value.

    Args:
        param_string: String like "x=0.5" or "app_name='drupe'" or just "value"

    Returns:
        Tuple of (parameter_name, parameter_value)

    Examples:
        >>> parse_single_parameter("x=0.5")
        ('x', 0.5)

        >>> parse_single_parameter("app_name='drupe'")
        ('app_name', 'drupe')

        >>> parse_single_parameter("'text'")
        ('arg_0', 'text')

        >>> parse_single_parameter("123")
        ('arg_0', 123)

        >>> parse_single_parameter("3")
        ('arg_0', 3)
    """
    # Pattern to match parameter name and value
    pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$"

    match = re.match(pattern, param_string)
    if match:
        # Named parameter
        param_name = match.group(1)
        param_value_str = match.group(2).strip()
        param_value = parse_value(param_value_str)
        return param_name, param_value
    else:
        # Positional parameter - treat as unnamed argument
        param_value = parse_value(param_string)
        return "arg_0", param_value


def parse_value(value_string: str) -> Any:
    """
    Parse a value string into appropriate Python type.

    Args:
        value_string: String representation of a value

    Returns:
        Parsed value (int, float, str, list, etc.)

    Examples:
        >>> parse_value("3")
        3

        >>> parse_value("3.14")
        3.14

        >>> parse_value("'hello'")
        'hello'

        >>> parse_value("[0.581, 0.898]")
        [0.581, 0.898]
    """
    value_string = value_string.strip()

    # String values (quoted)
    if (value_string.startswith("'") and value_string.endswith("'")) or (
        value_string.startswith('"') and value_string.endswith('"')
    ):
        return value_string[1:-1]

    # List values
    if value_string.startswith("[") and value_string.endswith("]"):
        return parse_list(value_string)

    # Dictionary values
    if value_string.startswith("{") and value_string.endswith("}"):
        return parse_dict(value_string)

    # Boolean values
    if value_string.lower() in ["true", "false"]:
        return value_string.lower() == "true"

    # None value
    if value_string.lower() == "none":
        return None

    # Numeric values
    try:
        # Try integer first
        if "." not in value_string:
            return int(value_string)
        else:
            return float(value_string)
    except ValueError:
        # If it's not a number, return as string (remove quotes if present)
        if value_string.startswith("'") and value_string.endswith("'"):
            return value_string[1:-1]
        elif value_string.startswith('"') and value_string.endswith('"'):
            return value_string[1:-1]
        else:
            return value_string


def parse_list(list_string: str) -> List[Any]:
    """
    Parse a list string into a Python list.

    Args:
        list_string: String like "[0.581, 0.898]"

    Returns:
        List of parsed values

    Examples:
        >>> parse_list("[0.581, 0.898]")
        [0.581, 0.898]
    """
    # Remove outer brackets
    content = list_string[1:-1].strip()
    if not content:
        return []

    # Split by commas, respecting nested structures
    parts = split_parameters(content)

    return [parse_value(part.strip()) for part in parts]


def parse_dict(dict_string: str) -> Dict[str, Any]:
    """
    Parse a dictionary string into a Python dict.

    Args:
        dict_string: String like "{'key': 'value'}"

    Returns:
        Dictionary of parsed key-value pairs
    """
    # Remove outer braces
    content = dict_string[1:-1].strip()
    if not content:
        return {}

    # Split by commas, respecting nested structures
    parts = split_parameters(content)

    result = {}
    for part in parts:
        part = part.strip()
        if ":" in part:
            key_str, value_str = part.split(":", 1)
            key = parse_value(key_str.strip())
            value = parse_value(value_str.strip())
            result[key] = value

    return result


def parse_multiple_functions(function_strings: List[str]) -> List[FunctionToolCall]:
    """
    Parse multiple function call strings.

    Args:
        function_strings: List of function call strings

    Returns:
        List of FunctionCall objects
    """
    results = []
    for func_str in function_strings:
        try:
            result_list = parse_function_call(func_str)
            results.extend(result_list)
        except Exception as e:
            print(f"Warning: Could not parse function call '{func_str}': {e}")
            continue

    return results


def extract_function_calls_from_text(text: str, block_regex: str = ".*") -> Tuple[str, List[FunctionToolCall]]:
    """
    Extract function calls from delimited code blocks inside *text*.

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
        ``(outside_text, function_calls)`` – the text with blocks stripped
        and the parsed function calls found inside the blocks.
    """
    func_pattern = r"[a-zA-Z_][a-zA-Z0-9_.]*\([^)]*\)"

    if block_regex:
        if not re.search(block_regex, text, flags=re.DOTALL):
            return text, []

        outside = "".join(re.split(block_regex, text, flags=re.DOTALL))
        blocks = re.findall(block_regex, text, flags=re.DOTALL)
        inside = " ".join(blocks)

        if not inside.strip():
            return outside, []

        matches = re.findall(func_pattern, inside)
        return outside, parse_multiple_functions(matches)

    return text, []