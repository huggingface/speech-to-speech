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
You have access to the following tools defined as Python functions:

{% for tool in tools %}
{{ tool.to_code_prompt() }}

{% endfor %}
When you need to use a tool, output the function call between \
{{ enter_code }} and {{ end_code }} tags in your response:

{{ enter_code }}
function_name(arg_name_1=value1, arg_name_2='string_value')
another_function(arg_name_1='hello')
{{ end_code }}

RULES:
- NEVER announce, introduce, or reference the tool call. \
Do NOT write "Here is the call", "I will use", "Let me call", \
"The function is", or any similar phrasing.
- The {{ enter_code }}…{{ end_code }} block must appear directly \
in your response without any surrounding explanation.
- Arguments MUST always be named: func(x=1, y=2). Positional arguments like func(1, 2) are NOT allowed.
- String arguments must be quoted: func(name='hello').
- Multiple tool calls can live in the same {{ enter_code }}…{{ end_code }} block.\
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
