"""Contract tests for the Realtime response lifecycle.

Every test in this module drives a scenario through the service and then
checks the emitted stream against two things:

1. The OpenAI SDK's own ``RealtimeServerEvent`` union, so each event has the
   shape the official clients parse.
2. ``assert_response_lifecycle_contract``, which encodes the ordering and
   status rules the Realtime API documents for
   ``response.output_item.added`` / ``.done``, ``response.content_part.added``
   / ``.done`` and ``response.done``.

The scenario matrix crosses output modality, tool-call timing, tool count,
whether TTS actually streams audio, and where the response ends, because the
lifecycle bugs this suite guards against only appear in specific combinations
of those.
"""

from typing import Any

import pytest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.pipeline.events import (
    AssistantOutputEvent,
    AssistantToolCallReadyEvent,
)
from speech_to_speech.pipeline.messages import (
    AssistantTextPart,
    AssistantToolCallPart,
)

from .realtime_contract import (
    assert_openai_schema,
    assert_response_lifecycle_contract,
)


def _pcm_bytes(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

_TEXT = AssistantTextPart
_TOOL = AssistantToolCallPart


def _tool(call_id: str) -> AssistantToolCallPart:
    return _TOOL(tool={"type": "function_call", "call_id": call_id, "name": "tool", "arguments": "{}"})


def _run_scenario(
    service: Any,
    conn_id: str,
    *,
    modalities: list[str],
    tool_timing: str,
    ending: str,
    audio: str = "streamed",
    tool_count: int = 1,
) -> list[Any]:
    """Drive one interleaving and return every event the response produced.

    ``ending`` picks where the response stops. ``complete`` and ``cancel`` run
    the whole script. ``cancel_midway`` cancels as soon as the first message is
    streamed, which for an early tool falls between the side channel and the
    ordered copy. ``cancel_after_first_tool`` cancels once the first ordered
    tool has closed but before the next item begins.
    """
    response_key = "response_1"
    wants_audio = "audio" in modalities and audio == "streamed"
    service._state(conn_id).current_response_params = RealtimeResponseCreateParams(
        output_modalities=modalities,
    )
    service.response._ensure_response(conn_id)

    collected: list[Any] = []
    tools = [_tool(f"c{n + 1}") for n in range(tool_count)]

    def cancel_now() -> list[Any]:
        return service.finish_response(conn_id, status="cancelled")

    if tool_timing == "early":
        # The side channel exposes the first tool before the ordered stream
        # reaches it. Any second tool follows in its own sequence slot.
        for offset, tool in enumerate(tools):
            collected += service.dispatch_pipeline_event(
                conn_id,
                AssistantToolCallReadyEvent(
                    response_key=response_key,
                    output_sequence=1 + offset,
                    part=tool,
                ),
            )

    collected += service.dispatch_pipeline_event(
        conn_id,
        AssistantOutputEvent(
            response_key=response_key,
            output_sequence=0,
            parts=[_TEXT(text="before")],
        ),
    )
    if wants_audio:
        collected += service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)

    if ending == "cancel_midway":
        return collected + cancel_now()

    if tool_timing != "none":
        for offset, tool in enumerate(tools):
            collected += service.dispatch_pipeline_event(
                conn_id,
                AssistantOutputEvent(
                    response_key=response_key,
                    output_sequence=1 + offset,
                    parts=[tool],
                ),
            )
            if ending == "cancel_after_first_tool" and offset == 0:
                return collected + cancel_now()
        collected += service.dispatch_pipeline_event(
            conn_id,
            AssistantOutputEvent(
                response_key=response_key,
                output_sequence=1 + len(tools),
                parts=[_TEXT(text="after")],
            ),
        )
        if wants_audio:
            collected += service.encode_audio_chunk(conn_id, _pcm_bytes(256), response_key)

    status = "cancelled" if ending == "cancel" else "completed"
    collected += service.finish_response(conn_id, status=status)
    return collected


@pytest.mark.parametrize("modalities", [["audio"], ["text"]], ids=["audio", "text_only"])
@pytest.mark.parametrize("tool_timing", ["none", "ordered", "early"])
@pytest.mark.parametrize("ending", ["complete", "cancel", "cancel_midway", "cancel_after_first_tool"])
def test_response_lifecycle_holds_across_interleavings(service, conn_id, modalities, tool_timing, ending):
    """Every modality, tool timing and ending must keep the same contract."""
    if ending == "cancel_after_first_tool" and tool_timing == "none":
        pytest.skip("no tool call to cancel after")
    events = _run_scenario(
        service,
        conn_id,
        modalities=modalities,
        tool_timing=tool_timing,
        ending=ending,
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(
        events,
        wants_audio="audio" in modalities,
        expected_status="completed" if ending == "complete" else "cancelled",
    )


@pytest.mark.parametrize("modalities", [["audio"], ["text"]], ids=["audio", "text_only"])
@pytest.mark.parametrize("ending", ["complete", "cancel"])
def test_response_done_reports_the_output_modalities(service, conn_id, modalities, ending):
    """response.done must report the modality the response was resolved to.

    A client reads this to know whether to expect audio at all, so it has to
    come from the same decision that selects the events, and agree with them.
    """
    events = _run_scenario(
        service,
        conn_id,
        modalities=modalities,
        tool_timing="none",
        ending=ending,
    )
    wants_audio = "audio" in modalities
    assert_openai_schema(events)
    assert_response_lifecycle_contract(
        events,
        wants_audio=wants_audio,
        expected_status="completed" if ending == "complete" else "cancelled",
    )
    done = next(event for event in events if event.type == "response.done")
    # The values are exclusive: "audio" already implies its transcript, and
    # OpenAI does not allow asking for both at once.
    assert done.response.output_modalities == (["audio"] if wants_audio else ["text"])

    streamed_audio = any(event.type.startswith("response.output_audio.") for event in events)
    streamed_text = any(event.type.startswith("response.output_text.") for event in events)
    assert streamed_audio == wants_audio, "audio events disagree with the reported modalities"
    assert streamed_text != wants_audio, "text events disagree with the reported modalities"


@pytest.mark.parametrize("modalities", [["audio"], ["text"]], ids=["audio", "text_only"])
@pytest.mark.parametrize("tool_timing", ["ordered", "early"])
@pytest.mark.parametrize("ending", ["complete", "cancel", "cancel_after_first_tool"])
def test_lifecycle_holds_for_two_tool_calls_in_a_row(service, conn_id, modalities, tool_timing, ending):
    """Back to back tool calls must each get their own item and close once."""
    events = _run_scenario(
        service,
        conn_id,
        modalities=modalities,
        tool_timing=tool_timing,
        ending=ending,
        tool_count=2,
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(
        events,
        wants_audio="audio" in modalities,
        expected_status="completed" if ending == "complete" else "cancelled",
    )
    done = next(event for event in events if event.type == "response.done")
    if ending == "cancel_after_first_tool":
        # The run stops between the two ordered tools. With early delivery the
        # side channel has already exposed both, so both items exist.
        expected_calls = ["c1", "c2"] if tool_timing == "early" else ["c1"]
        assert [item.type for item in done.response.output] == [
            "message",
            *["function_call"] * len(expected_calls),
        ]
        assert [item.call_id for item in done.response.output if item.type == "function_call"] == expected_calls
    else:
        assert [item.type for item in done.response.output] == [
            "message",
            "function_call",
            "function_call",
            "message",
        ]
        assert [item.call_id for item in done.response.output if item.type == "function_call"] == [
            "c1",
            "c2",
        ]


@pytest.mark.parametrize("tool_timing", ["none", "ordered", "early"])
@pytest.mark.parametrize("ending", ["complete", "cancel"])
def test_lifecycle_holds_when_tts_never_delivers_audio(service, conn_id, tool_timing, ending):
    """An audio response whose TTS produces no PCM must still close its items."""
    events = _run_scenario(
        service,
        conn_id,
        modalities=["audio"],
        tool_timing=tool_timing,
        ending=ending,
        audio="silent",
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(
        events,
        wants_audio=True,
        expected_status="completed" if ending == "complete" else "cancelled",
    )


def test_contract_checker_catches_an_unclosed_item(service, conn_id):
    """The checker must fail on a stream that drops an output_item.done.

    Without this, a checker that silently passes everything would give false
    confidence to the matrix above.
    """
    events = _run_scenario(
        service,
        conn_id,
        modalities=["text"],
        tool_timing="early",
        ending="cancel",
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(events, wants_audio=False)

    last_index = max(event.output_index for event in events if event.type == "response.output_item.added")
    without_last_close = [
        event
        for event in events
        if not (event.type == "response.output_item.done" and event.output_index == last_index)
    ]
    with pytest.raises(AssertionError, match="output items never closed"):
        assert_response_lifecycle_contract(without_last_close, wants_audio=False)


def test_contract_checker_catches_a_status_contradiction(service, conn_id):
    """The checker must fail when response.done contradicts output_item.done."""
    events = _run_scenario(
        service,
        conn_id,
        modalities=["text"],
        tool_timing="ordered",
        ending="cancel",
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(events, wants_audio=False)

    done = next(event for event in events if event.type == "response.done")
    flipped = done.response.output[0].model_copy(update={"status": "incomplete"})
    done.response.output[0] = flipped
    with pytest.raises(AssertionError, match="response.done says"):
        assert_response_lifecycle_contract(events, wants_audio=False)


def test_contract_checker_catches_a_content_part_closed_twice(service, conn_id):
    """The checker must fail when a content part is closed more than once."""
    events = _run_scenario(
        service,
        conn_id,
        modalities=["text"],
        tool_timing="none",
        ending="complete",
    )
    assert_openai_schema(events)
    assert_response_lifecycle_contract(events, wants_audio=False)

    part_done_index = next(index for index, event in enumerate(events) if event.type == "response.content_part.done")
    duplicated = [*events[: part_done_index + 1], events[part_done_index], *events[part_done_index + 1 :]]
    with pytest.raises(AssertionError, match="content part closed twice"):
        assert_response_lifecycle_contract(duplicated, wants_audio=False)
