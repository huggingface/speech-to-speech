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
- You may say one brief natural sentence before the tool call; for slow information tools, briefly say that you will check.
- For expression/background tools, always speak first. For requested expressions, use a short pattern like "Sure, here's my best <emotion>."; otherwise use a fitting empathetic sentence.
- Do not mention tags, functions, or tools. Keep prose outside tags brief, and do not claim tool results before a tool result is available.
- Use named arguments only; quote strings. Omit optional args instead of placeholder values like "random", "none", "", or null.
- Only one tool call may appear in a response.\
""",
    keep_trailing_newline=True,
)

# Text-channel variant: same call format and structural rules, without the
# voice-specific "speak first" prose.
TEXT_TOOL_PROMPT_TEMPLATE = Template(
    """\
Available tools:

{% for tool in tools %}
{{ tool.to_code_prompt() }}

{% endfor %}
To call a tool, put exactly one named-argument function call inside {{ enter_code }}...{{ end_code }}:
{{ enter_code }}function_name(required_arg='value'){{ end_code }}

Rules:
- Call a tool directly when it helps fulfill the request; no preamble sentence is required.
- Do not mention tags, functions, or tools in your prose, and do not claim tool results before a tool result is available.
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
    *,
    text_only: bool = False,
) -> str:
    """Render the tool-calling system-prompt section.

    Returns an empty string when *tools* is empty so it can be
    unconditionally appended to a base system prompt.  When *text_only* is set,
    the written-channel variant is used (no voice "speak first" prose).
    """
    if not tools:
        return ""

    template = TEXT_TOOL_PROMPT_TEMPLATE if text_only else TOOL_PROMPT_TEMPLATE
    return template.render(
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
