"""
Optional system-prompt builder that instructs a local LLM to output tool calls
inside delimited code blocks (e.g. ``<code>func(arg='val')</code>``).

The prompt is rendered from a Jinja2 template and relies on
``FunctionTool.to_code_prompt()`` to expose each tool's Python-style signature.
"""

import re

from jinja2 import Template

from speech_to_speech.LLM.tool_call.function_tool import FunctionTool

# ---------------------------------------------------------------------------
# Default delimiters
# ---------------------------------------------------------------------------
ENTER_CODE = "<code>"
END_CODE = "</code>"

# ---------------------------------------------------------------------------
# Jinja2 template
# ---------------------------------------------------------------------------
# ``enter_code`` / ``end_code`` are the block delimiters the model must wrap
# every tool call in.  ``tools`` is a list of FunctionTool instances whose
# ``.to_code_prompt()`` is called inside the template.
# ---------------------------------------------------------------------------

TOOL_PROMPT_TEMPLATE = Template(
    """\
Available tools:

{% for tool in tools %}
{{ tool.to_code_prompt() }}

{% endfor %}
To call a tool, put exactly one named-argument function call inside {{ enter_code }}...{{ end_code }}:
{{ enter_code }}function_name(required_arg='value'){{ end_code }}

Rules:
- You may say one brief natural sentence before the tool call, e.g. "Let me check with my camera."
- For expression/background tools, always speak first. If asked to show an expression, say something like "Sure, here's my best sadness."; otherwise say something like "That sounds really hard."
- Do not mention tags, functions, or tools. No prose inside the tags or after {{ end_code }}.
- Use named arguments only; quote strings. Omit optional args instead of placeholder values like "random", "none", "", or null.
- Only one tool call may appear in a response.\
""",
    keep_trailing_newline=True,
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_tool_system_prompt(
    tools: list[FunctionTool],
    enter_code: str = ENTER_CODE,
    end_code: str = END_CODE,
) -> str:
    """Render the tool-calling system-prompt section.

    Returns an empty string when *tools* is empty so it can be
    unconditionally appended to a base system prompt.
    """
    if not tools:
        return ""

    return TOOL_PROMPT_TEMPLATE.render(
        tools=tools,
        enter_code=enter_code,
        end_code=end_code,
    )


def build_block_regex(
    enter_code: str = ENTER_CODE,
    end_code: str = END_CODE,
) -> str:
    """Build a regex that matches a single code block (non-greedy).

    >>> build_block_regex("<code>", "</code>")
    '<code>.*?</code>'
    """
    return f"{re.escape(enter_code)}.*?{re.escape(end_code)}"
