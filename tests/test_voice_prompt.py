from speech_to_speech.LLM.tool_call.function_tool import FunctionTool
from speech_to_speech.LLM.tool_call.tool_prompt import build_tool_system_prompt
from speech_to_speech.LLM.voice_prompt import build_voice_system_prompt


def test_voice_prompt_makes_speech_the_default():
    prompt = build_voice_system_prompt("Be concise.")

    assert "Speech is the default." in prompt
    assert "Do not use tools for ordinary conversational behavior" in prompt
    assert "acknowledgments, greetings, agreement, listening" in prompt
    assert "include one brief spoken sentence in the same response" in prompt
    assert "call at most one tool" in prompt
    assert "Prefer a spoken response without tools when uncertain." in prompt


def test_voice_prompt_preserves_idle_action_escape_hatch():
    prompt = build_voice_system_prompt("Be concise.")

    assert "idle behavior" in prompt
    assert "idle action" in prompt


def test_local_tool_prompt_forbids_multiple_tool_calls():
    prompt = build_tool_system_prompt(
        [
            FunctionTool(
                type="function",
                name="dance",
                description="Dance once.",
                parameters={"type": "object", "properties": {}},
            )
        ]
    )

    assert "Only one tool call may appear in a response." in prompt
    assert "Multiple tool calls can live" not in prompt
