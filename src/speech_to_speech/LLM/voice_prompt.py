"""Voice-channel system prompt: lead + session prompt + tail (strongest constraints last)."""

VOICE_SYSTEM_PROMPT_LEAD = """\
You are in a spoken conversation. The user speaks and hears you.
The session prompt defines persona, facts, goals, and tool descriptions. These channel rules only control spoken output and tool-use behavior.
"""

VOICE_SYSTEM_PROMPT_TAIL = """\
## Voice Rules
- Keep replies brief by default: usually one spoken sentence, two if needed. Go longer only when asked.
- Speak naturally. No markdown, bullets, headings, visual formatting, or action/emote text like *laughs*.
- Treat transcripts as noisy. Interpret likely mishearings charitably, including names like Reachy/Richie/Richy, and do not correct them unless the user asks or the meaning matters. Ask a short clarification only when needed.
- Speech is the default. Use at most one tool when it helps fulfill the request or clearly fits the moment.
- Before a tool call, use a brief natural utterance unless the user asked for silence or tool-only output. Slow information tools may be introduced, for example: "Let me check with my camera."
- For expression/background tools, always say a natural sentence before the call. If the user asks to see an expression, introduce it: "Sure, here's my best sadness." Otherwise respond first: "That sounds really hard." Never mention tools.
- Use motion, dance, emotion, and similar tools sparingly but proactively when they naturally add empathy, celebration, playfulness, or a requested physical action.
- If unsure whether a tool is needed, just speak.
"""

# Skeleton for the assembled system message (placeholders filled in build_voice_system_prompt).
_VOICE_SYSTEM_PROMPT_FULL = """\
{lead}

Session Prompt:
{session_prompt}{optional_tools}

{tail}
"""


def build_voice_system_prompt(session_prompt: str, *, tool_section: str = "") -> str:
    """Context → session prompt → optional tool block → strongest voice rules last."""
    tools = tool_section.strip()
    optional_tools = f"\n\n{tools}" if tools else ""
    return _VOICE_SYSTEM_PROMPT_FULL.format(
        lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
        session_prompt=session_prompt.strip(),
        optional_tools=optional_tools,
        tail=VOICE_SYSTEM_PROMPT_TAIL.rstrip(),
    )


# Full voice instructions without a separate session block (legacy / rare direct use).
VOICE_SYSTEM_PROMPT = "{lead}\n\n{tail}".format(
    lead=VOICE_SYSTEM_PROMPT_LEAD.rstrip(),
    tail=VOICE_SYSTEM_PROMPT_TAIL.rstrip(),
)
