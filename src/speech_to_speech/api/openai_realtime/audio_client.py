from __future__ import annotations

import asyncio
import base64
import logging
import signal
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class RealtimeAudioClientConfig:
    """Configuration for the packaged microphone/speaker Realtime client."""

    url: str = "ws://127.0.0.1:8765/v1/realtime"
    model: str = "local"
    api_key: str = "test-key"
    send_rate: int = 16000
    recv_rate: int = 16000
    chunk_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    instructions: Optional[str] = None
    voice: Optional[str] = None
    print_json: bool = False
    block_mic_during_playback: bool = False
    connection_retry_timeout_s: float = 30.0


def normalize_realtime_url(url: str) -> tuple[str, str]:
    """Convert a full Realtime endpoint into the base URLs expected by the SDK."""

    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"ws", "wss", "http", "https"} or not parsed.netloc:
        raise ValueError("--url must be an absolute ws://, wss://, http://, or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("--url must not include a query string or fragment")

    path = parsed.path.rstrip("/")
    if not path.endswith("/realtime"):
        raise ValueError("--url must be the full Realtime endpoint ending in /realtime")

    sdk_path = path[: -len("/realtime")]
    websocket_scheme = "wss" if parsed.scheme in {"wss", "https"} else "ws"
    http_scheme = "https" if websocket_scheme == "wss" else "http"
    websocket_base_url = urlunsplit((websocket_scheme, parsed.netloc, sdk_path, "", ""))
    base_url = urlunsplit((http_scheme, parsed.netloc, sdk_path, "", ""))
    return base_url, websocket_base_url


def _make_client(config: RealtimeAudioClientConfig) -> AsyncOpenAI:
    base_url, websocket_base_url = normalize_realtime_url(config.url)
    return AsyncOpenAI(
        api_key=config.api_key,
        base_url=base_url,
        websocket_base_url=websocket_base_url,
    )


def build_session_update(config: RealtimeAudioClientConfig) -> dict[str, Any]:
    """Build the Realtime session update used by local and standalone clients."""

    def maybe_pcm_format(rate: int) -> Optional[dict[str, Any]]:
        # The OpenAI SDK models only validate explicit PCM formats at 24 kHz.
        # Omitting the format selects this server's native 16 kHz pipeline rate.
        if rate == 16000:
            return None
        if rate == 24000:
            return {"type": "audio/pcm", "rate": 24000}
        raise ValueError(
            f"Unsupported rate {rate}. Use 16000 for the local pipeline default "
            "or 24000 for the OpenAI Realtime PCM schema."
        )

    input_config: dict[str, Any] = {
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
    }
    output_config: dict[str, Any] = {}

    input_format = maybe_pcm_format(config.send_rate)
    output_format = maybe_pcm_format(config.recv_rate)
    if input_format is not None:
        input_config["format"] = input_format
    if output_format is not None:
        output_config["format"] = output_format
    if config.voice:
        output_config["voice"] = config.voice

    session: dict[str, Any] = {
        "type": "realtime",
        "audio": {
            "input": input_config,
            "output": output_config,
        },
    }
    if config.instructions:
        session["instructions"] = config.instructions
    return {"type": "session.update", "session": session}


class PlaybackBuffer:
    """Thread-safe audio state shared by the Realtime loop and sounddevice callbacks."""

    def __init__(self, recv_rate: int) -> None:
        self.recv_rate = recv_rate
        self._audio = bytearray()
        self._lock = Lock()
        self._active_until = 0.0

    def clear(self) -> None:
        with self._lock:
            self._active_until = 0.0
            self._audio.clear()

    def append(self, audio: bytes) -> None:
        with self._lock:
            self._audio.extend(audio)
            self._active_until = time.monotonic() + max(0.15, len(audio) / (2 * self.recv_rate))

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._audio) or time.monotonic() < self._active_until

    def write(self, outdata: Any) -> None:
        needed = len(outdata)
        with self._lock:
            available = min(needed, len(self._audio))
            if available:
                outdata[:available] = self._audio[:available]
                del self._audio[:available]
            if available < needed:
                outdata[available:] = b"\x00" * (needed - available)

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._audio)


class _FriendlyEventRenderer:
    def __init__(self) -> None:
        self.partial_user_text = ""
        self.live_user_width = 0
        self.saw_user_speech = False

    def render_live_user_text(self, text: str, *, final: bool = False) -> None:
        line = f"USER: {text}"
        padded = line + (" " * max(0, self.live_user_width - len(line)))
        if final:
            print(f"\r{padded}", flush=True)
            self.live_user_width = 0
            return
        print(f"\r{padded}", end="", flush=True)
        self.live_user_width = len(line)

    def clear_live_user_text(self) -> None:
        if self.live_user_width == 0:
            return
        print("\r" + (" " * self.live_user_width) + "\r", end="", flush=True)
        self.live_user_width = 0


def handle_server_event(
    event: Any,
    *,
    playback: PlaybackBuffer,
    renderer: _FriendlyEventRenderer,
    print_json: bool,
) -> None:
    """Apply one Realtime lifecycle event to local playback and console state."""

    if print_json:
        try:
            print(f"EVENT: {event.model_dump_json()}", flush=True)
        except Exception:
            print(f"EVENT: {event}", flush=True)

    if event.type == "session.created":
        print("Connected.", flush=True)
    elif event.type == "input_audio_buffer.speech_started":
        playback.clear()
        renderer.partial_user_text = ""
        if renderer.saw_user_speech:
            print("", flush=True)
        renderer.saw_user_speech = True
    elif event.type == "input_audio_buffer.speech_stopped":
        return
    elif event.type == "conversation.item.input_audio_transcription.delta":
        # This server emits the latest partial hypothesis, not a token suffix.
        renderer.partial_user_text = event.delta.strip()
        if renderer.partial_user_text:
            renderer.render_live_user_text(renderer.partial_user_text)
    elif event.type == "conversation.item.input_audio_transcription.completed":
        renderer.partial_user_text = ""
        renderer.render_live_user_text(event.transcript.strip(), final=True)
    elif event.type == "response.created":
        renderer.clear_live_user_text()
        print("ASSISTANT: <response started>", flush=True)
    elif event.type == "response.output_audio.delta":
        playback.append(base64.b64decode(event.delta))
    elif event.type == "response.output_audio.done":
        print("ASSISTANT: <audio done>", flush=True)
    elif event.type == "response.output_audio_transcript.done":
        print(f"ASSISTANT: {event.transcript}", flush=True)
    elif event.type == "response.function_call_arguments.done":
        print(
            f"TOOL: {event.name} call_id={event.call_id} arguments={event.arguments}",
            flush=True,
        )
    elif event.type == "response.done":
        if event.response.status == "cancelled":
            playback.clear()
        print(f"ASSISTANT: <response {event.response.status}>", flush=True)
    elif event.type == "output_audio_buffer.cleared":
        playback.clear()
    elif event.type == "error":
        renderer.clear_live_user_text()
        print(f"ERROR: {event.error.type}: {event.error.message}", flush=True)
    else:
        renderer.clear_live_user_text()
        print(f"EVENT: {event.type}", flush=True)


async def _wait_for_stop(stop_event: Event) -> None:
    while not stop_event.is_set():
        await asyncio.to_thread(stop_event.wait, 0.1)


async def _run_audio_session(
    conn: Any,
    config: RealtimeAudioClientConfig,
    stop_event: Event,
) -> None:
    import sounddevice as sd

    mic_queue: Queue[bytes] = Queue(maxsize=128)
    playback = PlaybackBuffer(config.recv_rate)
    renderer = _FriendlyEventRenderer()

    def callback_recv(outdata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Speaker status: %s", status)
        playback.write(outdata)

    def callback_send(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Microphone status: %s", status)
        if config.block_mic_during_playback and playback.is_active():
            return
        try:
            mic_queue.put_nowait(bytes(indata))
        except Full:
            logger.debug("Dropping local microphone chunk because the send queue is full")

    async def send_audio() -> None:
        while not stop_event.is_set():
            try:
                chunk = await asyncio.to_thread(mic_queue.get, True, 0.1)
            except Empty:
                continue
            await conn.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )

    async def receive_events() -> None:
        while not stop_event.is_set():
            event = await conn.recv()
            handle_server_event(
                event,
                playback=playback,
                renderer=renderer,
                print_json=config.print_json,
            )

    input_stream = sd.RawInputStream(
        samplerate=config.send_rate,
        channels=1,
        dtype="int16",
        blocksize=config.chunk_size,
        callback=callback_send,
        device=config.input_device,
    )
    output_stream = sd.RawOutputStream(
        samplerate=config.recv_rate,
        channels=1,
        dtype="int16",
        blocksize=config.chunk_size,
        callback=callback_recv,
        device=config.output_device,
    )

    input_stream.start()
    output_stream.start()
    try:
        tasks = {
            asyncio.create_task(send_audio()),
            asyncio.create_task(receive_events()),
            asyncio.create_task(_wait_for_stop(stop_event)),
        }

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        stop_event.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
    finally:
        stop_event.set()
        renderer.clear_live_user_text()
        input_stream.stop()
        output_stream.stop()
        input_stream.close()
        output_stream.close()


async def listen_and_play_realtime(
    config: RealtimeAudioClientConfig,
    *,
    stop_event: Event | None = None,
) -> None:
    """Connect microphone/speaker audio to a Realtime server over WebSocket."""

    owned_stop_event = stop_event is None
    stop_event = stop_event or Event()
    client = _make_client(config)
    connected = False
    retry_started = time.monotonic()

    try:
        while not stop_event.is_set():
            try:
                async with client.realtime.connect(model=config.model) as conn:
                    connected = True
                    await conn.send(build_session_update(config))  # type: ignore[arg-type]
                    await _run_audio_session(
                        conn,
                        config,
                        stop_event,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop_event.is_set():
                    return
                if connected or time.monotonic() - retry_started >= config.connection_retry_timeout_s:
                    raise
                logger.debug("Realtime loopback server is not ready yet: %s", exc)
                await asyncio.sleep(0.1)
    finally:
        if owned_stop_event:
            stop_event.set()
        await client.close()


def run_realtime_audio_client(config: RealtimeAudioClientConfig) -> None:
    """Run the audio client until SIGINT, SIGTERM, disconnect, or an error."""

    stop_event = Event()
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_shutdown(_sig: int, _frame: Any) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_shutdown)

    try:
        asyncio.run(listen_and_play_realtime(config, stop_event=stop_event))
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


class RealtimeAudioClient:
    """ThreadManager handler that embeds the packaged client for ``local``."""

    def __init__(self, stop_event: Event, config: RealtimeAudioClientConfig) -> None:
        self.stop_event = stop_event
        self.config = config

    def run(self) -> None:
        try:
            asyncio.run(
                listen_and_play_realtime(
                    self.config,
                    stop_event=self.stop_event,
                )
            )
        except Exception:
            logger.exception("Local Realtime audio client stopped unexpectedly")
            self.stop_event.set()
