from speech_to_speech.LLM.text_prompt import TEXT_SYSTEM_PROMPT, build_text_system_prompt
from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import build_tool_system_prompt


def test_text_prompt_keeps_persona_in_session_prompt():
    prompt = build_text_system_prompt("Be helpful.")

    assert "Be helpful." in prompt
    assert "You are a helpful assistant in a text conversation." in prompt
    assert "You are a helpful assistant in a text conversation." in TEXT_SYSTEM_PROMPT


def test_text_prompt_allows_markdown_and_drops_voice_rules():
    prompt = build_text_system_prompt("Be helpful.")

    assert "Use markdown when it helps" in prompt
    assert "If unsure whether a tool is needed, just answer directly." in prompt
    # No spoken-channel rules leak into the text prompt.
    assert "Speech is the default." not in prompt
    assert "Treat transcripts as noisy." not in prompt
    assert "Before a tool call, use a brief natural utterance" not in prompt
    assert "speak first" not in prompt


def _dance_tool() -> FunctionTool:
    return FunctionTool(
        type="function",
        name="dance",
        description="Dance once.",
        parameters={"type": "object", "properties": {}},
    )


def test_text_tool_prompt_drops_speak_first_but_keeps_structural_rules():
    prompt = build_tool_system_prompt([_dance_tool()], text_only=True)

    assert "no preamble sentence is required" in prompt
    assert "Only one tool call may appear in a response." in prompt
    assert "Omit optional args instead of placeholder values" in prompt
    assert "do not claim tool results before a tool result is available" in prompt
    # Voice choreography must not appear in the text variant.
    assert "always speak first" not in prompt
    assert "Sure, here's my best <emotion>." not in prompt
    assert "fitting empathetic sentence" not in prompt


def test_voice_tool_prompt_is_default():
    prompt = build_tool_system_prompt([_dance_tool()])

    assert "always speak first" in prompt
