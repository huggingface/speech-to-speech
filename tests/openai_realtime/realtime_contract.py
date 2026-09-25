"""Shared checks for the Realtime response lifecycle.

Two checks, used by the service-level contract tests and the transport tests:

1. ``assert_openai_schema`` and ``parse_wire_events`` validate events against
   the OpenAI SDK's own ``RealtimeServerEvent`` union, so each event has the
   shape the official clients parse.
2. ``assert_response_lifecycle_contract`` encodes the ordering and status
   rules the Realtime API documents for ``response.output_item.added`` /
   ``.done``, ``response.content_part.added`` / ``.done`` and
   ``response.done``.

This module is not collected by pytest; it only holds helpers.
"""

from typing import Any

from openai.types.realtime import RealtimeServerEvent
from pydantic import TypeAdapter

_SERVER_EVENT = TypeAdapter(RealtimeServerEvent)

# Events that carry a per-item payload keyed by output_index.
_ITEM_SCOPED = {
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
}

_CONTENT_EVENTS = {
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
}

# The content part type each streaming event belongs inside.
_EVENT_PART_TYPE = {
    "response.output_text.delta": "text",
    "response.output_text.done": "text",
    "response.output_audio.delta": "audio",
    "response.output_audio.done": "audio",
    "response.output_audio_transcript.delta": "audio",
    "response.output_audio_transcript.done": "audio",
}


def assert_openai_schema(events: list[Any]) -> None:
    """Every emitted event must parse as an official OpenAI server event."""
    for index, event in enumerate(events):
        payload = event.model_dump(mode="json", exclude_none=True)
        try:
            _SERVER_EVENT.validate_python(payload)
        except Exception as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"event {index} ({payload.get('type')}) does not match the OpenAI schema: {exc}"
            ) from exc


def parse_wire_events(messages: list[dict[str, Any]]) -> list[Any]:
    """Parse raw wire JSON with the OpenAI SDK's own models.

    Used by the transport tests so the lifecycle is asserted on the objects an
    official client would build, not on the server's internal types.
    """
    parsed: list[Any] = []
    for index, message in enumerate(messages):
        try:
            parsed.append(_SERVER_EVENT.validate_python(message))
        except Exception as exc:  # pragma: no cover - failure path
            raise AssertionError(
                f"wire message {index} ({message.get('type')}) is not a valid OpenAI server event: {exc}"
            ) from exc
    return parsed


def assert_response_lifecycle_contract(
    events: list[Any],
    *,
    wants_audio: bool,
    expected_status: str | None = None,
) -> None:
    """Check the documented ordering and status rules for one response.

    The rules, each phrased as the client-visible promise it protects:

    * an item announced with ``output_item.added`` is closed exactly once;
    * a message carries exactly one content part, opened before anything
      streams into it and closed before the item closes;
    * nothing is emitted for an item after it closes;
    * ``output_index`` is dense from zero and each index keeps one item id;
    * every event belongs to one response id, the one ``response.done`` reports;
    * the status in ``output_item.done`` survives into ``response.done``;
    * ``response.done`` reports exactly the items that were announced;
    * a text-only response closes each item before announcing the next one,
      because only pending TTS audio justifies leaving one open;
    * a content part keeps one ``content_index`` and one type from open to
      close, and only matching events stream inside it;
    * ``response.done`` reports ``expected_status`` when the caller gives one.

    ``wants_audio`` is required rather than read off the stream, so a wrong
    ``output_modalities`` on the response cannot quietly relax the rules.
    """
    added: dict[int, str] = {}
    done_status: dict[int, str] = {}
    done_items: dict[int, Any] = {}
    part_open: set[int] = set()
    part_closed: set[int] = set()
    part_content_index: dict[int, int] = {}
    part_type: dict[int, str] = {}
    item_ids: dict[int, str] = {}
    response_ids: set[str] = set()
    response_done: Any | None = None

    for event in events:
        etype = event.type
        if etype == "response.done":
            assert response_done is None, "response.done emitted more than once"
            response_done = event
            continue
        assert response_done is None, f"{etype} emitted after response.done"

        if etype == "response.created":
            response_ids.add(event.response.id)
            continue
        if etype not in _ITEM_SCOPED:
            continue
        index = event.output_index
        response_ids.add(event.response_id)

        if etype != "response.output_item.added":
            assert index in added, f"{etype} for output_index {index} before its output_item.added"
        assert index not in done_status, f"{etype} for output_index {index} after its output_item.done"

        item_id = getattr(event, "item_id", None) or getattr(getattr(event, "item", None), "id", None)
        if item_id is not None:
            assert item_ids.setdefault(index, item_id) == item_id, (
                f"output_index {index} changed item id from {item_ids[index]} to {item_id}"
            )

        if etype == "response.output_item.added":
            assert index not in added, f"output_index {index} announced twice"
            if not wants_audio:
                still_open = sorted(set(added) - set(done_status))
                assert not still_open, (
                    f"text-only response announced output_index {index} while "
                    f"{still_open} were still open; with no TTS to wait for, a "
                    "finished item must close at the boundary"
                )
            added[index] = event.item.type
        elif etype == "response.content_part.added":
            assert index not in part_open, f"content part opened twice on output_index {index}"
            part_open.add(index)
            part_content_index[index] = event.content_index
            part_type[index] = event.part.type
        elif etype == "response.content_part.done":
            assert index in part_open, f"content part closed without being opened on output_index {index}"
            assert index not in part_closed, f"content part closed twice on output_index {index}"
            assert event.content_index == part_content_index[index], (
                f"content part on output_index {index} opened at content_index "
                f"{part_content_index[index]} but closed at {event.content_index}"
            )
            assert event.part.type == part_type[index], (
                f"content part on output_index {index} opened as {part_type[index]} but closed as {event.part.type}"
            )
            part_closed.add(index)
        elif etype in _CONTENT_EVENTS:
            assert index in part_open, f"{etype} on output_index {index} outside an open content part"
            assert index not in part_closed, f"{etype} on output_index {index} after its content part closed"
            assert event.content_index == part_content_index[index], (
                f"{etype} on output_index {index} uses content_index {event.content_index}, "
                f"but the open part is content_index {part_content_index[index]}"
            )
            expected_part = _EVENT_PART_TYPE[etype]
            assert part_type[index] == expected_part, (
                f"{etype} on output_index {index} streams into a {part_type[index]} part, expected {expected_part}"
            )
        elif etype == "response.output_item.done":
            if added[index] == "message":
                assert index in part_open, f"message {index} closed without a content part"
                assert index in part_closed, f"message {index} closed before its content part"
            assert event.item.type == added[index], (
                f"output_index {index} was announced as {added[index]} but closed as {event.item.type}"
            )
            done_status[index] = event.item.status
            done_items[index] = event.item

    assert added, "no output items were announced"
    assert sorted(added) == list(range(len(added))), f"output indexes are not dense from zero: {sorted(added)}"
    unclosed = sorted(set(added) - set(done_status))
    assert not unclosed, f"output items never closed: {unclosed}"

    assert response_done is not None, "response.done was never emitted"
    response_ids.add(response_done.response.id)
    assert len(response_ids) == 1, f"events span more than one response id: {sorted(response_ids)}"
    if expected_status is not None:
        assert response_done.response.status == expected_status, (
            f"response.done reports status {response_done.response.status}, expected {expected_status}"
        )
    output = response_done.response.output
    assert len(output) == len(added), f"response.done reports {len(output)} items but {len(added)} were announced"
    for index, item in enumerate(output):
        assert item.id == item_ids[index], (
            f"response.done item {index} has id {item.id}, the stream used {item_ids[index]}"
        )
        assert item.status == done_status[index], (
            f"item {index} closed as {done_status[index]} but response.done says {item.status}"
        )
        assert item.type == added[index], (
            f"item {index} was announced as {added[index]} but response.done says {item.type}"
        )
        if item.type == "function_call":
            closed = done_items[index]
            assert (item.call_id, item.name, item.arguments) == (
                closed.call_id,
                closed.name,
                closed.arguments,
            ), (
                f"function call {index} changed between output_item.done and response.done: "
                f"{(closed.call_id, closed.name, closed.arguments)} became "
                f"{(item.call_id, item.name, item.arguments)}"
            )


class RealtimeClientHistory:
    """The user-turn history a generic Realtime client builds from events.

    It knows only standard event identity: items are keyed by ``item_id``,
    transcription deltas append to one item, and
    ``conversation.item.input_audio_transcription.completed`` carries that
    item's final transcript. It has no notion of a speculative revision, so
    every user item the server publishes here is permanent.
    """

    def __init__(self) -> None:
        self.speech_started: list[str] = []
        self.speech_stopped: list[str] = []
        self.deltas: dict[str, str] = {}
        self.transcripts: dict[str, str] = {}
        self.created_items: list[str] = []
        self.completed_items: list[str] = []
        self.failed_items: list[str] = []

    def apply(self, events: list[Any]) -> "RealtimeClientHistory":
        for event in events:
            etype = event.type
            if etype == "conversation.item.created" and getattr(event.item, "role", None) == "user":
                self.created_items.append(event.item.id)
            elif etype == "input_audio_buffer.speech_started":
                self.speech_started.append(event.item_id)
            elif etype == "input_audio_buffer.speech_stopped":
                self.speech_stopped.append(event.item_id)
            elif etype == "conversation.item.input_audio_transcription.delta":
                self.deltas[event.item_id] = self.deltas.get(event.item_id, "") + event.delta
            elif etype == "conversation.item.input_audio_transcription.completed":
                self.transcripts[event.item_id] = event.transcript
                self.completed_items.append(event.item_id)
            elif etype == "conversation.item.input_audio_transcription.failed":
                self.failed_items.append(event.item_id)
        return self

    @property
    def user_turns(self) -> list[str]:
        """Committed user transcripts, in conversation item order."""
        return [self.transcripts[item_id] for item_id in self.created_items if item_id in self.transcripts]


def assert_input_lifecycle_contract(events: list[Any]) -> None:
    """Check the documented lifecycle of one input audio item.

    The rules, each phrased as the client-visible promise it protects:

    * an item starts once, preserving its original audio onset;
    * speech stops on an item only after that item started;
    * a stop commits and creates the user item exactly once;
    * an item stops once, because ``speech_stopped`` announces the user
      message item created from that stopped buffer;
    * an item's transcription terminal arrives once and after its stop;
    * nothing streams into an item's transcript after that terminal.
    """
    started: set[str] = set()
    stopped: set[str] = set()
    terminal: set[str] = set()
    committed: dict[str, str | None] = {}
    created: set[str] = set()

    for event in events:
        etype = event.type
        item_id = getattr(event, "item_id", None)
        if etype == "input_audio_buffer.speech_started":
            assert item_id not in started, f"item {item_id} started more than once"
            started.add(item_id)
        elif etype == "input_audio_buffer.speech_stopped":
            assert item_id in started, f"speech stopped on item {item_id} that never started"
            assert item_id not in stopped, f"item {item_id} stopped more than once"
            stopped.add(item_id)
        elif etype == "input_audio_buffer.committed":
            assert item_id in stopped, f"item {item_id} committed before its stop"
            assert item_id not in committed, f"item {item_id} committed more than once"
            committed[item_id] = event.previous_item_id
        elif etype == "conversation.item.created" and event.item.id in started:
            item_id = event.item.id
            assert item_id in committed, f"item {item_id} created before commitment"
            assert item_id not in created, f"item {item_id} created more than once"
            assert event.previous_item_id == committed[item_id]
            assert event.previous_item_id != item_id
            assert getattr(event.item, "role", None) == "user"
            assert event.item.content[0].type == "input_audio"
            created.add(item_id)
        elif etype == "conversation.item.input_audio_transcription.delta":
            assert item_id not in terminal, f"transcript delta on item {item_id} after its terminal"
        elif etype in (
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        ):
            assert item_id not in terminal, f"item {item_id} terminalized more than once"
            if item_id in started:
                assert item_id in created, f"item {item_id} transcribed before it was created"
            terminal.add(item_id)

    assert stopped == set(committed) == created, "stopped input items must be committed and created"
