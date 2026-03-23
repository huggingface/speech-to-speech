"""Unit tests for RuntimeConfig.apply_session_update / _apply_update.

Verifies the merge semantics: only explicitly-set fields (model_fields_set)
are written, nested BaseModels recurse without clobbering siblings, and
explicit None clears a previously-set value.
"""

from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from openai.types.realtime.session_update_event import SessionUpdateEvent

from api.openai_realtime.runtime_config import RuntimeConfig


def _parse_session(**session_fields) -> RealtimeSessionCreateRequest:
    """Parse a session dict the same way the SDK does (tracks model_fields_set)."""
    session_fields.setdefault("type", "realtime")
    evt = SessionUpdateEvent.model_validate({
        "type": "session.update",
        "session": session_fields,
    })
    return evt.session


class TestApplySessionUpdate:

    def test_partial_update_preserves_untouched_fields(self):
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            instructions="Be a pirate",
            audio={"output": {"voice": "coral"}},
            tool_choice="auto",
        ))
        assert cfg.session.instructions == "Be a pirate"
        assert cfg.session.audio.output.voice == "coral"
        assert cfg.session.tool_choice == "auto"

        cfg.apply_session_update(_parse_session(instructions="Be an astronaut"))
        assert cfg.session.instructions == "Be an astronaut"
        assert cfg.session.audio.output.voice == "coral"
        assert cfg.session.tool_choice == "auto"

    def test_explicit_none_clears_field(self):
        """Sending ``"turn_detection": null`` clears a previously-set value."""
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            audio={"input": {"turn_detection": {"type": "server_vad"}}},
        ))
        assert cfg.session.audio.input.turn_detection is not None
        assert cfg.session.audio.input.turn_detection.type == "server_vad"

        cfg.apply_session_update(_parse_session(
            audio={"input": {"turn_detection": None}},
        ))
        assert cfg.session.audio.input.turn_detection is None

    def test_nested_sibling_preserved(self):
        """Updating audio.output.voice must not touch audio.input.turn_detection."""
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            audio={
                "input": {"turn_detection": {"type": "server_vad", "threshold": 0.6}},
                "output": {"voice": "echo"},
            },
        ))
        assert cfg.session.audio.output.voice == "echo"
        assert cfg.session.audio.input.turn_detection.threshold == 0.6

        cfg.apply_session_update(_parse_session(
            audio={"output": {"voice": "shimmer"}},
        ))
        assert cfg.session.audio.output.voice == "shimmer"
        assert cfg.session.audio.input.turn_detection.type == "server_vad"
        assert cfg.session.audio.input.turn_detection.threshold == 0.6

    def test_sequential_updates_accumulate(self):
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(instructions="Step 1"))
        cfg.apply_session_update(_parse_session(audio={"output": {"voice": "alloy"}}))
        cfg.apply_session_update(_parse_session(tool_choice="required"))

        assert cfg.session.instructions == "Step 1"
        assert cfg.session.audio.output.voice == "alloy"
        assert cfg.session.tool_choice == "required"

    def test_deep_nested_leaf_update(self):
        """Changing only turn_detection.threshold preserves the rest."""
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            audio={
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "silence_duration_ms": 800,
                    },
                },
            },
        ))
        assert cfg.session.audio.input.turn_detection.threshold == 0.5
        assert cfg.session.audio.input.turn_detection.silence_duration_ms == 800

        cfg.apply_session_update(_parse_session(
            audio={"input": {"turn_detection": {"type": "server_vad", "threshold": 0.8}}},
        ))
        assert cfg.session.audio.input.turn_detection.threshold == 0.8
        assert cfg.session.audio.input.turn_detection.silence_duration_ms == 800

    def test_tools_replaced_wholesale(self):
        """Tools is a list, not a BaseModel — the whole list is replaced."""
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            tools=[{"type": "function", "name": "get_weather"}],
        ))
        assert len(cfg.session.tools) == 1

        cfg.apply_session_update(_parse_session(
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "function", "name": "get_time"},
            ],
        ))
        assert len(cfg.session.tools) == 2

    def test_update_after_fresh_init(self):
        """apply_session_update works on a freshly-constructed RuntimeConfig."""
        cfg = RuntimeConfig()
        cfg.apply_session_update(_parse_session(
            instructions="Hello",
            audio={"output": {"voice": "sage"}},
        ))
        assert cfg.session.instructions == "Hello"
        assert cfg.session.audio.output.voice == "sage"

    def test_validator_ensures_audio_structure_on_init(self):
        """Default construction guarantees audio.input and audio.output are not None."""
        cfg = RuntimeConfig()
        assert cfg.session.audio is not None
        assert cfg.session.audio.input is not None
        assert cfg.session.audio.output is not None
