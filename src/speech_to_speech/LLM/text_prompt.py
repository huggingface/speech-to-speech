"""Text-channel system prompt: lead + session prompt + tail (strongest constraints last)."""

from speech_to_speech.LLM.utils import format_language_instruction

TEXT_SYSTEM_PROMPT_LEAD = """\
You are a helpful assistant in a text conversation.
"""

TEXT_SYSTEM_PROMPT_TAIL = """\
## Text Rules
- Write clearly and directly. Match length to the request: concise for simple questions, fuller when the task genuinely needs it.
- Use markdown when it helps (lists, code blocks, tables, emphasis); don't over-format simple answers.
- This is a written channel: no spoken-style filler and no action/emote text like *laughs*.
- Use tools when they help fulfill the request. No preamble sentence is required before a tool call.
- For slow or external tools, just call the tool and use the result; you don't need to announce it.
- If unsure whether a tool is needed, just answer directly.
"""

# Skeleton for the assembled system message (placeholders filled in build_text_system_prompt).
_TEXT_SYSTEM_PROMPT_FULL = """\
{lead}

Session Prompt:
{session_prompt}{optional_tools}{optional_language}

{tail}
"""


def build_text_system_prompt(
    session_prompt: str,
    *,
    tool_section: str = "",
    language_name: str | None = None,
) -> str:
    """Context → session prompt → optional tool block → optional language hint → text rules."""
    tools = tool_section.strip()
    optional_tools = f"\n\n{tools}" if tools else ""
    optional_language = ""
    if language_name:
        optional_language = f"\n\n{format_language_instruction(language_name)}"
    return _TEXT_SYSTEM_PROMPT_FULL.format(
        lead=TEXT_SYSTEM_PROMPT_LEAD.rstrip(),
        session_prompt=session_prompt.strip(),
        optional_tools=optional_tools,
        optional_language=optional_language,
        tail=TEXT_SYSTEM_PROMPT_TAIL.rstrip(),
    )


# Full text instructions without a separate session block (legacy / rare direct use).
TEXT_SYSTEM_PROMPT = "{lead}\n\n{tail}".format(
    lead=TEXT_SYSTEM_PROMPT_LEAD.rstrip(),
    tail=TEXT_SYSTEM_PROMPT_TAIL.rstrip(),
)
