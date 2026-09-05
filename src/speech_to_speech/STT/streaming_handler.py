from __future__ import annotations

import base64
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import monotonic, perf_counter
from typing import Any, Callable, Iterator, Literal, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import soxr

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.log_context import pipeline_log_ctx
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
OPENAI_REALTIME_BASE_URL = "wss://api.openai.com/v1"


class StreamingTranscriptionError(RuntimeError):
    """A sanitized stateful-transcription failure safe for client output."""


class _WebSocket(Protocol):
    def send(self, message: str) -> None: ...

    def recv(self, timeout: float | None = None) -> str | bytes: ...

    def close(self) -> None: ...


ConnectFactory = Callable[..., _WebSocket]


@dataclass(frozen=True)
class _ProtocolEvent:
    kind: Literal["ignore", "committed", "delta", "completed", "failed", "error"]
    text: str = ""
    language: str | None = None
    message: str = ""
    item_id: str | None = None
    content_index: int | None = None


class StreamingSTTProtocol(Protocol):
    name: str
    requires_session_updated: bool

    def session_update(self) -> dict[str, Any]: ...

    def start_utterance(self) -> dict[str, Any] | None: ...

    def append_audio(self, audio: bytes) -> dict[str, Any]: ...

    def finish_utterance(self) -> dict[str, Any]: ...

    def discard_utterance(self) -> dict[str, Any] | None: ...

    def parse_event(self, event: dict[str, Any]) -> _ProtocolEvent: ...


@dataclass(frozen=True)
class OpenAIRealtimeProtocol:
    """OpenAI Realtime transcription-session wire dialect."""

    model: str
    language: str | None
    audio_sample_rate: int

    name = "openai-realtime"
    requires_session_updated = True

    def session_update(self) -> dict[str, Any]:
        transcription: dict[str, Any] = {"model": self.model}
        if self.language:
            if self.model in {"gpt-live-transcribe", "gpt-transcribe"}:
                transcription["languages"] = [self.language]
            else:
                transcription["language"] = self.language
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.audio_sample_rate},
                        "transcription": transcription,
                        "turn_detection": None,
                    }
                },
            },
        }

    def start_utterance(self) -> dict[str, Any] | None:
        return None

    def append_audio(self, audio: bytes) -> dict[str, Any]:
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode("ascii"),
        }

    def finish_utterance(self) -> dict[str, Any]:
        return {"type": "input_audio_buffer.commit"}

    def discard_utterance(self) -> dict[str, Any] | None:
        return {"type": "input_audio_buffer.clear"}

    def parse_event(self, event: dict[str, Any]) -> _ProtocolEvent:
        event_type = event.get("type")
        raw_item_id = event.get("item_id")
        item_id = raw_item_id if isinstance(raw_item_id, str) and raw_item_id else None
        if event_type == "input_audio_buffer.committed":
            if item_id is None:
                return _ProtocolEvent("ignore")
            return _ProtocolEvent("committed", item_id=item_id)
        content_index = event.get("content_index")
        if event_type == "conversation.item.input_audio_transcription.delta" and "content_index" not in event:
            # Deltas may omit the index; this adapter sends one audio part.
            content_index = 0
        if event_type in {
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        } and (
            item_id is None
            or isinstance(content_index, bool)
            or not isinstance(content_index, int)
            or content_index < 0
        ):
            return _ProtocolEvent("ignore")
        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta")
            return _ProtocolEvent(
                "delta",
                text=delta if isinstance(delta, str) else "",
                item_id=item_id,
                content_index=content_index,
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript")
            language = _first_language_code(event.get("languages"))
            if language is None and isinstance(event.get("language"), str):
                language = event["language"]
            return _ProtocolEvent(
                "completed",
                text=transcript if isinstance(transcript, str) else "",
                language=language,
                item_id=item_id,
                content_index=content_index,
            )
        if event_type == "conversation.item.input_audio_transcription.failed":
            return _ProtocolEvent(
                "failed",
                message=_remote_error_message(event),
                item_id=item_id,
                content_index=content_index,
            )
        if event_type == "error":
            return _ProtocolEvent("error", message=_remote_error_message(event))
        return _ProtocolEvent("ignore")


@dataclass(frozen=True)
class VLLMRealtimeProtocol:
    """vLLM's distinct, experimental Realtime transcription wire dialect."""

    model: str
    language: str | None
    audio_sample_rate: int

    name = "vllm-realtime"
    requires_session_updated = False

    def session_update(self) -> dict[str, Any]:
        return {"type": "session.update", "model": self.model}

    def start_utterance(self) -> dict[str, Any] | None:
        return {"type": "input_audio_buffer.commit", "final": False}

    def append_audio(self, audio: bytes) -> dict[str, Any]:
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode("ascii"),
        }

    def finish_utterance(self) -> dict[str, Any]:
        return {"type": "input_audio_buffer.commit", "final": True}

    def discard_utterance(self) -> dict[str, Any] | None:
        # vLLM currently has no per-buffer clear event. Closing the socket is
        # the only way to guarantee rejected PCM cannot reach the next turn.
        return None

    def parse_event(self, event: dict[str, Any]) -> _ProtocolEvent:
        event_type = event.get("type")
        if event_type == "transcription.delta":
            delta = event.get("delta")
            return _ProtocolEvent("delta", text=delta if isinstance(delta, str) else "")
        if event_type == "transcription.done":
            text = event.get("text")
            return _ProtocolEvent("completed", text=text if isinstance(text, str) else "")
        if event_type == "error":
            return _ProtocolEvent("error", message=_remote_error_message(event))
        return _ProtocolEvent("ignore")


@dataclass(frozen=True)
class _AppendAudio:
    generation: int
    audio: bytes


@dataclass(frozen=True)
class _StartTurn:
    generation: int
    turn_id: str | None
    turn_revision: int | None


@dataclass
class _Commit:
    generation: int
    turn_id: str | None
    turn_revision: int | None
    done: Event
    boundary_queued_at_s: float
    result: str | None = None
    language: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Reset:
    generation: int


@dataclass(frozen=True)
class _DiscardAudio:
    generation: int


@dataclass(frozen=True)
class _Stop:
    pass


_Command = _AppendAudio | _StartTurn | _Commit | _DiscardAudio | _Reset | _Stop
_CommitKey = tuple[int, str | None, int | None]


def _first_language_code(value: object) -> str | None:
    if not isinstance(value, list):
        return None
    for language in value:
        if isinstance(language, str):
            return language
        if isinstance(language, dict) and isinstance(language.get("code"), str):
            return language["code"]
    return None


def _remote_error_message(event: dict[str, Any]) -> str:
    error = event.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(error, str):
        return error
    return "remote transcription error"


def _join_transcripts(prefix: str, text: str) -> str:
    prefix = prefix.strip()
    text = text.strip()
    if not prefix:
        return text
    if not text:
        return prefix
    return f"{prefix} {text}"


def _default_connect(url: str, *, headers: dict[str, str], open_timeout: float) -> _WebSocket:
    from websockets.sync.client import connect

    return connect(url, additional_headers=headers, open_timeout=open_timeout, close_timeout=1.0)


def _endpoint_url(base_url: str, model: str, *, include_model_query: bool) -> str:
    split = urlsplit(base_url.strip())
    scheme = {"http": "ws", "https": "wss"}.get(split.scheme, split.scheme)
    if scheme not in {"ws", "wss"} or not split.netloc:
        raise ValueError("Streaming STT base_url must be an http(s) or ws(s) URL")
    path = split.path.rstrip("/")
    if not path.endswith("/realtime"):
        path = f"{path}/realtime"
    query = dict(parse_qsl(split.query, keep_blank_values=True))
    # Hosted OpenAI transcription sessions select the session type with
    # intent=transcription. A transcription model in the query is rejected as
    # the realtime session model.
    if split.hostname == "api.openai.com":
        query.setdefault("intent", "transcription")
        query.pop("model", None)
    elif include_model_query:
        query.setdefault("model", model)
    return urlunsplit((scheme, split.netloc, path, urlencode(query), ""))


class _StreamingPCMResampler:
    """Preserve filter and phase state until one local utterance boundary."""

    def __init__(self, source_rate: int, target_rate: int) -> None:
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._stream = self._new_stream()

    def _new_stream(self) -> soxr.ResampleStream:
        return soxr.ResampleStream(
            self.source_rate,
            self.target_rate,
            1,
            dtype="int16",
        )

    def push(self, audio: bytes) -> bytes:
        samples = np.frombuffer(audio, dtype=np.int16)
        converted = self._stream.resample_chunk(samples, last=False)
        return converted.tobytes()

    def finish_utterance(self) -> bytes:
        converted = self._stream.resample_chunk(np.empty(0, dtype=np.int16), last=True)
        self._stream = self._new_stream()
        return converted.tobytes()

    def reset(self) -> None:
        self._stream = self._new_stream()


class _StreamingSession:
    """Own one reconnectable WebSocket and serialize its audio/commit commands."""

    def __init__(
        self,
        *,
        protocol: StreamingSTTProtocol,
        endpoint_url: str,
        headers: dict[str, str],
        connect_timeout: float,
        final_timeout: float,
        connect_factory: ConnectFactory,
        queue_out: Queue[Any],
        stop_event: Event,
        speculative_turns: SpeculativeTurnTracker | None,
        pipeline_index: int | None,
    ) -> None:
        self.protocol = protocol
        self.endpoint_url = endpoint_url
        self.headers = headers
        self.connect_timeout = connect_timeout
        self.final_timeout = final_timeout
        self.connect_factory = connect_factory
        self.queue_out = queue_out
        self.stop_event = stop_event
        self.speculative_turns = speculative_turns
        self.pipeline_index = pipeline_index

        self._commands: Queue[_Command] = Queue()
        self._generation_lock = Lock()
        self._generation = 0
        self._commit_lock = Lock()
        self._pending_commits: dict[_CommitKey, deque[_Commit]] = {}
        self._publication_lock = Lock()
        self._connection_lock = Lock()
        self._connection: _WebSocket | None = None
        self._thread = Thread(target=self._run, name=f"{protocol.name}-session", daemon=True)
        self._thread.start()

    @property
    def generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def append_audio(self, audio: bytes) -> None:
        if not audio or self.stop_event.is_set():
            return
        self._commands.put(_AppendAudio(self.generation, audio))

    def start_turn(self, turn_id: str | None, turn_revision: int | None) -> None:
        if self.stop_event.is_set():
            return
        self._commands.put(_StartTurn(self.generation, turn_id, turn_revision))

    def discard_utterance(self) -> None:
        if self.stop_event.is_set():
            return
        self._commands.put(_DiscardAudio(self.generation))

    def begin_commit(self, turn_id: str | None, turn_revision: int | None) -> None:
        generation = self.generation
        commit = _Commit(
            generation=generation,
            turn_id=turn_id,
            turn_revision=turn_revision,
            done=Event(),
            boundary_queued_at_s=perf_counter(),
        )
        key = (generation, turn_id, turn_revision)
        with self._commit_lock:
            self._pending_commits.setdefault(key, deque()).append(commit)
        self._commands.put(commit)

    def has_pending_commit(self, source: VADAudio) -> bool:
        key = (self.generation, source.turn_id, source.turn_revision)
        with self._commit_lock:
            return bool(self._pending_commits.get(key))

    def commit(self, source: VADAudio) -> _Commit | None:
        generation = self.generation
        key = (generation, source.turn_id, source.turn_revision)
        with self._commit_lock:
            pending = self._pending_commits.get(key)
            commit = pending.popleft() if pending else None
            if pending is not None and not pending:
                del self._pending_commits[key]
        if commit is None:
            commit = _Commit(
                generation=generation,
                turn_id=source.turn_id,
                turn_revision=source.turn_revision,
                done=Event(),
                boundary_queued_at_s=perf_counter(),
            )
            self._commands.put(commit)
        deadline = commit.boundary_queued_at_s + self.final_timeout
        while not commit.done.wait(timeout=min(0.05, max(0.0, deadline - perf_counter()))):
            if self.stop_event.is_set() or generation != self.generation:
                return None
            if perf_counter() >= deadline:
                # Setup or socket I/O may occupy the worker. Return promptly;
                # the worker still owns expiry, late-event rejection, and close.
                commit.error = "streaming transcription timed out"
                break
        if generation != self.generation:
            return None
        return commit

    def cancel_session(self) -> None:
        with self._generation_lock:
            self._generation += 1
            generation = self._generation
        # A publisher that already passed its generation check must finish
        # before cancellation returns and SESSION_END can be forwarded.
        with self._publication_lock:
            pass
        with self._commit_lock:
            self._pending_commits.clear()
        self._close_connection()
        self._commands.put(_Reset(generation))

    def close(self) -> None:
        self.cancel_session()
        self._commands.put(_Stop())
        self._thread.join(timeout=2.0)
        self._close_connection()

    def _run(self) -> None:
        if self.pipeline_index is not None:
            pipeline_log_ctx.set(self.pipeline_index)
        connection: _WebSocket | None = None
        worker_generation = self.generation
        turn_id: str | None = None
        turn_revision: int | None = None
        utterance_started = False
        utterance_has_audio = False
        remote_hypothesis = ""
        audio_error: str | None = None
        audio_error_requires_close = False
        active_commit: _Commit | None = None
        active_item_id: str | None = None
        active_content_index: int | None = None
        retired_item_ids: set[str] = set()
        committed_prefixes: dict[str, str] = {}
        failed_turn_ids: set[str | None] = set()
        failed_turn_message: str | None = None
        pending_unassigned_audio: deque[_AppendAudio] = deque()
        deferred_commands: deque[_StartTurn | _AppendAudio | _Commit | _DiscardAudio] = deque()

        def reset_utterance() -> None:
            nonlocal utterance_started, utterance_has_audio, remote_hypothesis, turn_id, turn_revision
            nonlocal active_item_id, active_content_index
            utterance_started = False
            utterance_has_audio = False
            remote_hypothesis = ""
            turn_id = None
            turn_revision = None
            active_item_id = None
            active_content_index = None

        def poison_turn(failed_turn_id: str | None, message: str) -> None:
            nonlocal failed_turn_message
            failed_turn_ids.clear()
            failed_turn_ids.add(failed_turn_id)
            failed_turn_message = message

        def clear_failed_turn() -> None:
            nonlocal failed_turn_message
            failed_turn_ids.clear()
            failed_turn_message = None

        def retire_active_item() -> None:
            if active_item_id is not None:
                retired_item_ids.add(active_item_id)

        def close_connection() -> None:
            nonlocal connection
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    logger.debug("Ignoring error while closing %s STT connection", self.protocol.name, exc_info=True)
            connection = None
            retired_item_ids.clear()
            self._publish_connection(None)

        def fail_commit(commit: _Commit, message: str) -> None:
            commit.error = message
            commit.done.set()

        def fail_connection(exc: BaseException, message: str = "streaming transcription connection failed") -> None:
            nonlocal active_commit, audio_error, audio_error_requires_close
            logger.warning("%s STT connection failed: %s", self.protocol.name, type(exc).__name__)
            close_connection()
            if active_commit is not None:
                poison_turn(active_commit.turn_id, message)
                fail_commit(active_commit, message)
                active_commit = None
                reset_utterance()
                audio_error = None
                audio_error_requires_close = False
            elif utterance_started or utterance_has_audio:
                poison_turn(turn_id, message)
                audio_error = message
                audio_error_requires_close = True

        def discard_utterance() -> None:
            nonlocal audio_error, audio_error_requires_close
            pending_unassigned_audio.clear()
            if connection is not None and (utterance_started or utterance_has_audio):
                discard_event = self.protocol.discard_utterance()
                if discard_event is None or active_item_id is None:
                    # Clearing bytes cannot fence late transcripts for an item
                    # whose identity we have not learned yet.
                    close_connection()
                else:
                    try:
                        send(discard_event)
                    except Exception as exc:
                        logger.warning(
                            "%s STT discard failed: %s",
                            self.protocol.name,
                            type(exc).__name__,
                        )
                        close_connection()
            retire_active_item()
            reset_utterance()
            audio_error = None
            audio_error_requires_close = False
            # Rejecting an unassigned fragment may clear its failure, but a
            # known turn must still fail if it reopens with missing audio.
            if None in failed_turn_ids:
                clear_failed_turn()

        def send(event: dict[str, Any]) -> None:
            if connection is None:
                raise StreamingTranscriptionError("streaming transcription is not connected")
            connection.send(json.dumps(event, separators=(",", ":")))

        def receive_setup_event(expected: str) -> None:
            if connection is None:
                raise StreamingTranscriptionError("streaming transcription is not connected")
            deadline = monotonic() + self.connect_timeout
            while monotonic() < deadline:
                try:
                    raw = connection.recv(timeout=min(0.05, deadline - monotonic()))
                except (TimeoutError, Empty):
                    continue
                event = self._decode_event(raw)
                if event.get("type") == "error":
                    raise StreamingTranscriptionError(_remote_error_message(event))
                if event.get("type") == expected:
                    return
            raise StreamingTranscriptionError(f"streaming transcription setup timed out waiting for {expected}")

        def ensure_connection() -> bool:
            nonlocal connection
            if connection is not None:
                return True
            setup_started_at_s = perf_counter()
            try:
                connection = self.connect_factory(
                    self.endpoint_url,
                    headers=self.headers,
                    open_timeout=self.connect_timeout,
                )
                self._publish_connection(connection)
                receive_setup_event("session.created")
                send(self.protocol.session_update())
                if self.protocol.requires_session_updated:
                    receive_setup_event("session.updated")
            except Exception as exc:
                fail_connection(exc)
                return False
            logger.info(
                "%s STT connection setup completed in %.3fs",
                self.protocol.name,
                perf_counter() - setup_started_at_s,
            )
            return True

        def combined_hypothesis(text: str) -> str:
            prefix = committed_prefixes.get(turn_id, "") if turn_id is not None else ""
            return _join_transcripts(prefix, text)

        def emit_partial() -> None:
            with self._publication_lock:
                if not remote_hypothesis or turn_id is None or not self._generation_is_current(worker_generation):
                    return
                output = PartialTranscription(
                    text=combined_hypothesis(remote_hypothesis),
                    turn_id=turn_id,
                    turn_revision=turn_revision,
                )
                tracker = self.speculative_turns
                if tracker is not None and not tracker.is_latest(output.turn_id, output.turn_revision):
                    return
                self.queue_out.put(output)

        def handle_protocol_event(event: _ProtocolEvent) -> None:
            nonlocal active_commit, active_item_id, active_content_index, remote_hypothesis
            nonlocal audio_error, audio_error_requires_close
            if active_commit is not None and perf_counter() >= active_commit.boundary_queued_at_s + self.final_timeout:
                fail_connection(TimeoutError(), "streaming transcription timed out")
                return
            if event.kind == "ignore":
                return

            if event.kind == "committed":
                if active_commit is None or event.item_id is None or event.item_id in retired_item_ids:
                    logger.debug(
                        "Ignoring retired or unowned %s STT event for item=%s", self.protocol.name, event.item_id
                    )
                    return
                if active_item_id is None:
                    active_item_id = event.item_id
                elif event.item_id != active_item_id:
                    logger.debug("Ignoring unowned %s STT event for item=%s", self.protocol.name, event.item_id)
                    return
                return

            if event.item_id is not None:
                if (
                    event.item_id in retired_item_ids
                    or (active_commit is None and not utterance_has_audio)
                    or (event.kind == "completed" and active_commit is None)
                ):
                    logger.debug(
                        "Ignoring retired or unowned %s STT event for item=%s", self.protocol.name, event.item_id
                    )
                    return
                if active_item_id is None:
                    active_item_id = event.item_id
                    active_content_index = event.content_index
                elif event.item_id != active_item_id:
                    logger.debug(
                        "Ignoring unowned %s STT event for item=%s content=%s",
                        self.protocol.name,
                        event.item_id,
                        event.content_index,
                    )
                    return
                elif active_content_index is None:
                    active_content_index = event.content_index
                elif event.content_index != active_content_index:
                    logger.debug(
                        "Ignoring unowned %s STT event for item=%s content=%s",
                        self.protocol.name,
                        event.item_id,
                        event.content_index,
                    )
                    return
            elif event.kind in {"delta", "failed"} and active_commit is None and not utterance_has_audio:
                logger.debug("Ignoring unowned %s STT event without an item ID", self.protocol.name)
                return
            if event.kind == "delta":
                if not event.text:
                    return
                remote_hypothesis += event.text
                emit_partial()
                return
            if event.kind == "error":
                logger.warning("%s STT remote error: %s", self.protocol.name, event.message)
                has_unfinished_turn = active_commit is not None or utterance_started or utterance_has_audio
                failed_turn_id = active_commit.turn_id if active_commit is not None else turn_id
                close_connection()
                message = "remote streaming transcription failed"
                if has_unfinished_turn:
                    poison_turn(failed_turn_id, message)
                if active_commit is not None:
                    fail_commit(active_commit, message)
                    active_commit = None
                    reset_utterance()
                    audio_error = None
                    audio_error_requires_close = False
                elif has_unfinished_turn:
                    audio_error = message
                    audio_error_requires_close = True
                return
            if event.kind == "failed":
                logger.warning(
                    "%s STT transcription failed for item=%s",
                    self.protocol.name,
                    event.item_id or "unknown",
                )
                message = "remote streaming transcription failed"
                failed_turn_id = active_commit.turn_id if active_commit is not None else turn_id
                poison_turn(failed_turn_id, message)
                retire_active_item()
                if active_commit is not None:
                    fail_commit(active_commit, message)
                    active_commit = None
                    audio_error = None
                else:
                    close_connection()
                    audio_error = message
                audio_error_requires_close = False
                reset_utterance()
                return
            if active_commit is None:
                logger.debug("Ignoring duplicate or unowned %s STT completion", self.protocol.name)
                return
            retire_active_item()
            combined = combined_hypothesis(event.text)
            if active_commit.turn_id is not None:
                committed_prefixes[active_commit.turn_id] = combined
            active_commit.result = combined
            active_commit.language = event.language
            logger.info(
                "%s VAD commit to final transcript completed in %.3fs turn=%s rev=%s",
                self.protocol.name,
                perf_counter() - active_commit.boundary_queued_at_s,
                active_commit.turn_id,
                active_commit.turn_revision,
            )
            active_commit.done.set()
            active_commit = None
            reset_utterance()
            audio_error = None
            audio_error_requires_close = False

        while True:
            # VAD may drop a stale final before process() consumes its commit.
            # The worker owns the deadline and releases deferred audio itself.
            if active_commit is not None and perf_counter() >= active_commit.boundary_queued_at_s + self.final_timeout:
                fail_connection(TimeoutError(), "streaming transcription timed out")
            command: _Command | None = None
            if active_commit is None and deferred_commands:
                command = deferred_commands.popleft()
            else:
                try:
                    command = self._commands.get(timeout=0.01)
                except Empty:
                    pass

            if isinstance(command, _Stop):
                if active_commit is not None:
                    fail_commit(active_commit, "streaming transcription stopped")
                close_connection()
                return

            if isinstance(command, _Reset):
                worker_generation = command.generation
                if active_commit is not None:
                    active_commit.done.set()
                    active_commit = None
                audio_error = None
                audio_error_requires_close = False
                committed_prefixes.clear()
                clear_failed_turn()
                pending_unassigned_audio.clear()
                deferred_commands.clear()
                reset_utterance()
                close_connection()
                continue

            if command is not None and getattr(command, "generation", worker_generation) != self.generation:
                if isinstance(command, _Commit):
                    command.done.set()
                continue

            # A provider connection can have only one unscoped transcription
            # in flight. Preserve later-turn commands locally until the active
            # final result settles so they cannot mutate or contaminate it.
            if active_commit is not None and isinstance(command, (_StartTurn, _AppendAudio, _Commit, _DiscardAudio)):
                deferred_commands.append(command)
                command = None

            if isinstance(command, _DiscardAudio):
                discard_utterance()

            if isinstance(command, _StartTurn):
                if failed_turn_ids:
                    if None in failed_turn_ids and command.turn_id is not None:
                        failed_turn_ids.clear()
                        failed_turn_ids.add(command.turn_id)
                    elif command.turn_id not in failed_turn_ids:
                        clear_failed_turn()
                        audio_error = None
                        audio_error_requires_close = False
                turn_id = command.turn_id
                turn_revision = command.turn_revision
                if command.turn_id in failed_turn_ids:
                    audio_error = failed_turn_message or "streaming transcription failed"
                    pending_unassigned_audio.clear()
                else:
                    while pending_unassigned_audio:
                        deferred_commands.appendleft(pending_unassigned_audio.pop())
                emit_partial()

            elif isinstance(command, _AppendAudio):
                if failed_turn_ids and turn_id is None:
                    pending_unassigned_audio.append(command)
                elif audio_error is None:
                    if not ensure_connection():
                        audio_error = "streaming transcription connection failed"
                        audio_error_requires_close = True
                        poison_turn(turn_id, audio_error)
                    else:
                        try:
                            if not utterance_started:
                                start_event = self.protocol.start_utterance()
                                if start_event is not None:
                                    send(start_event)
                                utterance_started = True
                            send(self.protocol.append_audio(command.audio))
                            utterance_has_audio = True
                        except Exception as exc:
                            utterance_has_audio = True
                            fail_connection(exc)

            elif isinstance(command, _Commit):
                turn_id = command.turn_id
                turn_revision = command.turn_revision
                if perf_counter() >= command.boundary_queued_at_s + self.final_timeout:
                    active_commit = command
                    fail_connection(TimeoutError(), "streaming transcription timed out")
                elif audio_error is not None:
                    should_close = audio_error_requires_close
                    fail_commit(command, audio_error)
                    audio_error = None
                    audio_error_requires_close = False
                    reset_utterance()
                    if should_close:
                        close_connection()
                elif not utterance_has_audio or connection is None:
                    fail_commit(command, "streaming transcription received no audio")
                    reset_utterance()
                else:
                    try:
                        active_commit = command
                        send(self.protocol.finish_utterance())
                    except Exception as exc:
                        fail_connection(exc)

            if connection is None:
                continue
            try:
                # Do not make queued PCM wait behind a receive timeout. Once a
                # final commit is active, a short blocking read avoids polling.
                raw = connection.recv(timeout=0.01 if active_commit is not None else 0.0)
            except (TimeoutError, Empty):
                continue
            except Exception as exc:
                fail_connection(exc)
                continue
            try:
                event = self._decode_event(raw)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                fail_connection(exc)
                continue
            handle_protocol_event(self.protocol.parse_event(event))

    def _generation_is_current(self, generation: int) -> bool:
        return generation == self.generation and not self.stop_event.is_set()

    def _publish_connection(self, connection: _WebSocket | None) -> None:
        with self._connection_lock:
            self._connection = connection

    def _close_connection(self) -> None:
        with self._connection_lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                logger.debug("Ignoring error while interrupting %s STT connection", self.protocol.name, exc_info=True)

    @staticmethod
    def _decode_event(raw: str | bytes) -> dict[str, Any]:
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        if not isinstance(payload, dict):
            raise ValueError("streaming transcription event must be an object")
        return payload


class StatefulStreamingSTTHandler(BaseSTTHandler):
    """STT handler that pairs incremental PCM ingress with local VAD commits."""

    protocol_type: type[OpenAIRealtimeProtocol] | type[VLLMRealtimeProtocol]
    include_model_query = False
    experimental = False

    def setup(
        self,
        base_url: str,
        model: str,
        audio_sample_rate: int,
        connect_timeout: float,
        final_timeout: float,
        api_key: str | None = None,
        language: str | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        connect_factory: ConnectFactory | None = None,
        pipeline_index: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Streaming STT requires a model")
        if audio_sample_rate <= 0:
            raise ValueError("Streaming STT audio_sample_rate must be > 0")
        if connect_timeout <= 0 or final_timeout <= 0:
            raise ValueError("Streaming STT timeouts must be > 0")
        if self.protocol_type is VLLMRealtimeProtocol and audio_sample_rate != PIPELINE_SAMPLE_RATE:
            raise ValueError("vLLM Realtime STT requires 16 kHz PCM")

        self.speculative_turns = speculative_turns
        self.final_revision_settle_s = 0.0
        self.audio_sample_rate = audio_sample_rate
        self._audio_lock = Lock()
        self._resampler = (
            _StreamingPCMResampler(PIPELINE_SAMPLE_RATE, audio_sample_rate)
            if audio_sample_rate != PIPELINE_SAMPLE_RATE
            else None
        )
        protocol = self.protocol_type(
            model=model.strip(),
            language=language.strip() if language else None,
            audio_sample_rate=audio_sample_rate,
        )
        endpoint_url = _endpoint_url(
            base_url,
            model.strip(),
            include_model_query=self.include_model_query,
        )
        normalized_endpoint = urlsplit(endpoint_url)._replace(query="", fragment="").geturl()
        if api_key is None and normalized_endpoint == f"{OPENAI_REALTIME_BASE_URL}/realtime":
            api_key = os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._session = _StreamingSession(
            protocol=protocol,
            endpoint_url=endpoint_url,
            headers=headers,
            connect_timeout=connect_timeout,
            final_timeout=final_timeout,
            connect_factory=connect_factory or _default_connect,
            queue_out=self.queue_out,
            stop_event=self.stop_event,
            speculative_turns=speculative_turns,
            pipeline_index=pipeline_index,
        )
        if self.experimental:
            logger.warning("%s protocol is experimental", protocol.name)

    def append_audio(self, audio: bytes) -> None:
        with self._audio_lock:
            if self._resampler is not None:
                audio = self._resampler.push(audio)
            self._session.append_audio(audio)

    def start_turn(self, turn_id: str | None, turn_revision: int | None) -> None:
        self._session.start_turn(turn_id, turn_revision)

    def commit_boundary(self, turn_id: str | None, turn_revision: int | None) -> None:
        with self._audio_lock:
            if self._resampler is not None:
                self._session.append_audio(self._resampler.finish_utterance())
            self._session.begin_commit(turn_id, turn_revision)

    def discard_utterance(self) -> None:
        with self._audio_lock:
            if self._resampler is not None:
                self._resampler.reset()
            self._session.discard_utterance()

    def cancel_session(self) -> None:
        with self._audio_lock:
            if self._resampler is not None:
                self._resampler.reset()
            self._session.cancel_session()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        if vad_audio.mode == "progressive":
            return
        if not self._session.has_pending_commit(vad_audio):
            self.commit_boundary(vad_audio.turn_id, vad_audio.turn_revision)
        generation = self._session.generation
        commit = self._session.commit(vad_audio)
        if commit is None or generation != self._session.generation:
            return
        if commit.error is not None:
            logger.error(
                "%s STT failed turn=%s rev=%s: %s",
                self.__class__.__name__,
                vad_audio.turn_id,
                vad_audio.turn_revision,
                commit.error,
            )
            yield TranscriptionFailure(
                message=commit.error,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
            return
        if self.speculative_turns is not None and not self.speculative_turns.is_latest(
            vad_audio.turn_id,
            vad_audio.turn_revision,
        ):
            return
        yield Transcription(
            text=commit.result or "",
            language_code=commit.language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def on_session_end(self) -> None:
        self.cancel_session()
        super().on_session_end()

    def cleanup(self) -> None:
        self._session.close()


class OpenAIRealtimeSTTHandler(StatefulStreamingSTTHandler):
    protocol_type = OpenAIRealtimeProtocol
    include_model_query = True


class VLLMRealtimeSTTHandler(StatefulStreamingSTTHandler):
    protocol_type = VLLMRealtimeProtocol
    experimental = True
