"""Voice-channel system prompt: lead + session prompt + tail (strongest constraints last)."""

from speech_to_speech.LLM.utils import format_language_instruction

VOICE_SYSTEM_PROMPT_LEAD = """\
You are in a spoken conversation. The user speaks and hears you.
The session prompt defines persona, facts, goals, and tool descriptions. These channel rules only control spoken output and tool-use behavior.
"""

VOICE_SYSTEM_PROMPT_TAIL = """\
## Voice Rules
- Keep replies brief by default: usually one spoken sentence, two if needed. Go longer only when asked.
- Speak naturally. No markdown, bullets, headings, visual formatting, or action/emote text like *laughs*.
- Treat transcripts as noisy. Correct likely mishearings only if asked or meaning depends on it.
- Speech is the default. Use tools when they help fulfill the request or fit the moment.
- Never mention tools or their function names in spoken output.
- For information tools, act immediately rather than merely offering. You may give one brief acknowledgement before the first call. After tool results, make further calls without speaking. Once you have enough results, give one final answer; do not narrate individual calls.
- For expression/background tools, speak first. If asked to show an expression, use a short pattern like "Sure, here's my best <emotion>." Otherwise use a fitting empathetic sentence.
- After completed expression/background/physical-action tools, do not add a second spoken comment unless the result has user-facing information.
- Use motion, dance, emotion, and similar tools sparingly when they add empathy, celebration, playfulness, or a requested physical action.
- If unsure whether a tool is needed, just speak.
"""

# Skeleton for the assembled system message (placeholders filled in build_voice_system_prompt).
_VOICE_SYSTEM_PROMPT_FULL = """\
{lead}

Session Prompt:
{session_prompt}{optional_tools}{optional_language}

{tail}
"""


def build_voice_system_prompt(
    session_prompt: str,
    *,
    tool_section: str = "",
    language_name: str | None = None,
) -> str:
    """Context → session prompt → optional tool block → optional language hint → voice rules."""
    tools = tool_section.strip()
    optional_tools = f"\n\n{tools}" if tools else ""
    optional_language = ""
    if language_name:
        optional_language = f"\n\n{format_language_instruction(language_name)}"
    return _VOICE_SYSTEM_PROMPT_FULL.format(
        lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
        session_prompt=session_prompt.strip(),
        optional_tools=optional_tools,
        optional_language=optional_language,
        tail=VOICE_SYSTEM_PROMPT_TAIL.rstrip(),
    )


# Full voice instructions without a separate session block (legacy / rare direct use).
VOICE_SYSTEM_PROMPT = "{lead}\n\n{tail}".format(
    lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
    tail=VOICE_SYSTEM_PROMPT_TAIL.rstrip(),
)
