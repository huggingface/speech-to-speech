"""Tests for the WebRTC transport.

Three layers:

- Pure-unit tests for the shared ``append_pcm`` path, the ``PcmResampler``
  (stereo downmix, statefulness), and the paced ``PipelineAudioTrack``.
- Dispatch tests driving ``_dispatch_client_event`` with a fake transport,
  covering the transport-gated events (append rejected over WebRTC,
  output_audio_buffer.clear flushing server-side audio).
- One loopback integration test: a real aiortc peer performs the SDP
  handshake against the uvicorn-served app (POST /v1/realtime/calls),
  exchanges events over the 'oai-events' data channel, streams mic audio
  into the pipeline input queue, and receives paced audio from output_queue.

The whole module is skipped when the ``webrtc`` extra (aiortc) isn't installed.
"""

import asyncio
import json
import time
from queue import Empty, Queue
from threading import Event as ThreadingEvent

import numpy as np
import pytest

aiortc = pytest.importorskip("aiortc")
av = pytest.importorskip("av")

import httpx  # noqa: E402  (ships with the openai dependency)
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

import speech_to_speech.api.openai_realtime.websocket_router as router_module  # noqa: E402
from speech_to_speech.api.openai_realtime.pipeline_unit import PipelineUnit  # noqa: E402
from speech_to_speech.api.openai_realtime.service import CHUNK_SIZE_BYTES, RealtimeService  # noqa: E402
from speech_to_speech.api.openai_realtime.transports import SessionTransport  # noqa: E402
from speech_to_speech.api.openai_realtime.webrtc_session import (  # noqa: E402
    WEBRTC_FRAME_SAMPLES,
    WEBRTC_SAMPLE_RATE,
    PcmResampler,
    PipelineAudioTrack,
)
from speech_to_speech.pipeline.cancel_scope import CancelScope  # noqa: E402
from speech_to_speech.pipeline.events import SpeechStartedEvent  # noqa: E402
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE  # noqa: E402

from .test_openai_client import _ServerEnv  # noqa: E402

PIPELINE_SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit() -> PipelineUnit:
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    return PipelineUnit(
        index=0,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=Queue(),
        text_output_queue=Queue(),
        text_prompt_queue=text_prompt_queue,
        handlers=[],
    )


class _FakeTransport(SessionTransport):
    kind = "webrtc"

    def __init__(self):
        self.sent: list[dict] = []
        self.discards = 0

    async def send_events(self, events):
        self.sent.extend(e.model_dump() for e in events)

    async def send_audio_chunk(self, service, session_id, pcm):
        raise AssertionError("dispatch tests never send audio")

    def discard_pending_audio(self):
        self.discards += 1

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# append_pcm (shared inbound path)
# ---------------------------------------------------------------------------


class TestAppendPcm:
    def test_chunks_and_remainder_carry_across_calls(self):
        unit = _make_unit()
        conn_id = unit.service.register()

        # 700 samples at pipeline rate: one 512-sample chunk + 188 remainder.
        chunks = unit.service.append_pcm(conn_id, b"\x01\x00" * 700, PIPELINE_SAMPLE_RATE)
        assert [len(c) for c in chunks] == [CHUNK_SIZE_BYTES]

        # 324 more completes the second chunk exactly (188 + 324 = 512).
        chunks = unit.service.append_pcm(conn_id, b"\x01\x00" * 324, PIPELINE_SAMPLE_RATE)
        assert [len(c) for c in chunks] == [CHUNK_SIZE_BYTES]
        assert unit.service._state(conn_id).audio_remainder == b""

    def test_sets_commit_bookkeeping(self):
        unit = _make_unit()
        conn_id = unit.service.register()

        assert unit.service.handle_audio_commit(conn_id) is not None  # empty buffer errors

        unit.service.append_pcm(conn_id, b"\x01\x00" * 512, PIPELINE_SAMPLE_RATE)
        assert unit.service.handle_audio_commit(conn_id) is None


# ---------------------------------------------------------------------------
# PcmResampler
# ---------------------------------------------------------------------------


class TestPcmResampler:
    def test_stereo_48k_downmixes_to_mono_16k(self):
        resampler = PcmResampler(PIPELINE_SAMPLE_RATE)
        n = WEBRTC_FRAME_SAMPLES
        stereo = np.zeros((2, n), dtype=np.int16)
        stereo[0, :] = 1000
        stereo[1, :] = 3000
        frame = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = 0

        total = bytearray(resampler.resample_frame(frame))
        # Push several frames so filter delay flushes through.
        for i in range(1, 10):
            f = av.AudioFrame.from_ndarray(stereo, format="s16p", layout="stereo")
            f.sample_rate = WEBRTC_SAMPLE_RATE
            f.pts = i * n
            total += resampler.resample_frame(f)

        samples = np.frombuffer(bytes(total), dtype=np.int16)
        # 10 frames of 20 ms at 48 kHz → ~200 ms at 16 kHz = ~3200 samples
        # (minus filter delay). A plane-concatenating flatten bug would give
        # double that; a channel-dropping bug would average to 1000 or 3000.
        assert 2800 <= samples.shape[0] <= 3200
        steady_state = samples[samples.shape[0] // 2 :]
        assert abs(int(np.mean(steady_state)) - 2000) <= 10  # downmix average

    def test_stateful_across_pcm_chunks(self):
        resampler = PcmResampler(WEBRTC_SAMPLE_RATE)
        total = bytearray()
        for _ in range(10):
            total += resampler.resample_pcm(b"\x01\x00" * 512, PIPELINE_SAMPLE_RATE)
        samples = np.frombuffer(bytes(total), dtype=np.int16)
        # 5120 samples at 16 kHz → ~15360 at 48 kHz, minus filter delay.
        assert 15000 <= samples.shape[0] <= 15360


# ---------------------------------------------------------------------------
# PipelineAudioTrack
# ---------------------------------------------------------------------------


class TestPipelineAudioTrack:
    async def test_recv_returns_written_audio_then_silence(self):
        track = PipelineAudioTrack()
        payload = (np.ones(WEBRTC_FRAME_SAMPLES, dtype=np.int16) * 5).tobytes()
        track.write(payload)

        frame = await track.recv()
        assert frame.sample_rate == WEBRTC_SAMPLE_RATE
        assert np.all(frame.to_ndarray() == 5)

        frame = await track.recv()  # buffer now empty → silence
        assert np.all(frame.to_ndarray() == 0)
        track.stop()

    async def test_recv_paces_to_wall_clock(self):
        track = PipelineAudioTrack()
        track.write(b"\x00" * WEBRTC_FRAME_SAMPLES * 2 * 10)  # 10 frames buffered

        start = time.monotonic()
        for _ in range(5):
            await track.recv()
        elapsed = time.monotonic() - start
        # 5 frames of 20 ms: first is immediate, the rest paced → ≥ ~80 ms.
        # Without pacing this loop completes in microseconds.
        assert elapsed >= 0.06
        track.stop()

    async def test_clear_drops_unplayed_audio(self):
        track = PipelineAudioTrack()
        track.write((np.ones(WEBRTC_FRAME_SAMPLES * 4, dtype=np.int16) * 7).tobytes())
        assert track.buffered_bytes > 0

        track.clear()
        assert track.buffered_bytes == 0
        frame = await track.recv()
        assert np.all(frame.to_ndarray() == 0)
        track.stop()

    async def test_recv_after_stop_raises(self):
        track = PipelineAudioTrack()
        track.stop()
        with pytest.raises(MediaStreamError):
            await track.recv()


# ---------------------------------------------------------------------------
# Client-event dispatch over the data channel
# ---------------------------------------------------------------------------


class TestWebRTCDispatch:
    async def test_append_rejected_over_webrtc(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "input_audio_buffer.append", "audio": "AAAA"},
            transport,
            transport_kind="webrtc",
        )

        assert len(transport.sent) == 1
        assert transport.sent[0]["type"] == "error"
        assert transport.sent[0]["error"]["type"] == "invalid_event_for_transport"
        assert unit.input_queue.qsize() == 0

    async def test_output_audio_buffer_clear_flushes_audio(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        unit.output_queue.put(b"\x01\x00" * 512)
        unit.output_queue.put(AUDIO_RESPONSE_DONE)

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "output_audio_buffer.clear"},
            transport,
            transport_kind="webrtc",
        )

        assert transport.sent == []  # no error
        assert transport.discards == 1
        # Pending audio flushed, done-sentinel preserved so the response still closes.
        assert unit.output_queue.get_nowait() == AUDIO_RESPONSE_DONE
        with pytest.raises(Empty):
            unit.output_queue.get_nowait()

    async def test_output_audio_buffer_clear_rejected_over_websocket(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()
        transport.kind = "websocket"

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "output_audio_buffer.clear"},
            transport,
            transport_kind="websocket",
        )

        assert len(transport.sent) == 1
        assert transport.sent[0]["type"] == "error"
        assert transport.sent[0]["error"]["type"] == "invalid_event_for_transport"

    async def test_response_cancel_discards_transport_audio(self):
        unit = _make_unit()
        conn_id = unit.service.register()
        transport = _FakeTransport()

        await router_module._dispatch_client_event(
            unit,
            conn_id,
            {"type": "response.cancel"},
            transport,
            transport_kind="webrtc",
        )

        assert transport.discards == 1


# ---------------------------------------------------------------------------
# Send-loop barge-in against transport-buffered audio
# ---------------------------------------------------------------------------


class TestBargeInAfterResponseDone:
    """Speech starting after a response finished must still flush audio the
    transport buffered but has not played yet: finish_response() runs when the
    done-sentinel is observed, not when playback completes, so fast TTS can
    leave seconds of unplayed audio in the WebRTC track with in_response
    already cleared."""

    def test_speech_start_flushes_buffered_transport_audio(self):
        unit = _make_unit()
        stop_event = ThreadingEvent()
        app = router_module.create_app(pool=[unit], stop_event=stop_event)
        with TestClient(app) as client:
            with client.websocket_connect("/v1/realtime") as ws:
                ws.receive_json()  # session.created
                # No response is active or pending. Swap in a spy transport so
                # the send loop's discard call is observable.
                spy = _FakeTransport()
                assert unit.session is not None
                unit.session.transport = spy
                generation_before = unit.cancel_scope.generation

                unit.text_output_queue.put(SpeechStartedEvent())

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and spy.discards == 0:
                    time.sleep(0.02)
                assert spy.discards == 1
                # Nothing to cancel: no response was active.
                assert unit.cancel_scope.generation == generation_before
        stop_event.set()


# ---------------------------------------------------------------------------
# Loopback integration: real aiortc peer against the served app
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env():
    env = _ServerEnv()
    env.start()
    yield env
    env.stop()


class _DataChannelInbox:
    """Collects data-channel messages and lets tests await specific types."""

    def __init__(self, dc):
        self.events: list[dict] = []
        self._new = asyncio.Event()

        @dc.on("message")
        def on_message(msg):
            self.events.append(json.loads(msg))
            self._new.set()

    async def wait_for(self, event_type: str, timeout: float = 5.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            for event in self.events:
                if event["type"] == event_type:
                    return event
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(
                    f"No '{event_type}' event within {timeout}s; got {[e['type'] for e in self.events]}"
                )
            self._new.clear()
            try:
                await asyncio.wait_for(self._new.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass


class TestWebRTCLoopback:
    async def test_handshake_events_and_audio_roundtrip(self, server_env):
        pc = RTCPeerConnection()
        try:
            dc = pc.createDataChannel("oai-events")
            inbox = _DataChannelInbox(dc)
            pc.addTrack(AudioStreamTrack())  # silent mic track

            received_frames: list = []
            track_ready = asyncio.Event()

            @pc.on("track")
            def on_track(track):
                async def _consume():
                    while True:
                        try:
                            frame = await track.recv()
                        except MediaStreamError:
                            return
                        received_frames.append(frame)
                        track_ready.set()

                asyncio.ensure_future(_consume())

            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert resp.status_code == 201
            assert resp.headers["content-type"].startswith("application/sdp")
            assert resp.headers["location"].startswith("/v1/realtime/calls/")

            await pc.setRemoteDescription(RTCSessionDescription(sdp=resp.text, type="answer"))

            # session.created arrives once the data channel opens.
            created = await inbox.wait_for("session.created", timeout=10.0)
            assert "session" in created

            # Client events over the data channel reach the shared dispatch:
            # append must be rejected as transport-invalid.
            dc.send(json.dumps({"type": "input_audio_buffer.append", "audio": "AAAA"}))
            error = await inbox.wait_for("error")
            assert error["error"]["type"] == "invalid_event_for_transport"

            # Inbound mic audio lands on input_queue as 512-sample chunks.
            def _wait_for_input_chunk(timeout: float = 10.0):
                return server_env.input_queue.get(timeout=timeout)

            chunk, _cfg = await asyncio.get_running_loop().run_in_executor(None, _wait_for_input_chunk)
            assert len(chunk) == CHUNK_SIZE_BYTES

            # Outbound pipeline audio: PCM through output_queue reaches the
            # client as RTP frames, and the done-sentinel closes the response
            # over the data channel (response.created was sent implicitly).
            server_env.output_queue.put(np.ones(4096, dtype=np.int16).tobytes())
            server_env.output_queue.put(AUDIO_RESPONSE_DONE)

            await inbox.wait_for("response.created", timeout=10.0)
            done = await inbox.wait_for("response.done", timeout=10.0)
            assert done["response"]["status"] == "completed"

            await asyncio.wait_for(track_ready.wait(), timeout=10.0)
            assert received_frames[0].sample_rate == WEBRTC_SAMPLE_RATE

            # Hanging up: closing the data channel signals the server (an
            # SCTP reset, unlike a bare pc.close() which the server only
            # notices via ICE consent timeouts). The release path enqueues
            # SESSION_END and marks the session as draining.
            dc.close()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                session = server_env.unit.session
                if session is None or session.released_at is not None:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("WebRTC disconnect did not start the unit release")
        finally:
            await pc.close()

    async def test_rejects_wrong_content_type(self, server_env):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                json={"sdp": "nope"},
                timeout=10.0,
            )
        assert resp.status_code == 415

    async def test_rejects_when_pool_full(self, server_env):
        pc = RTCPeerConnection()
        try:
            pc.createDataChannel("oai-events")
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                first = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
                assert first.status_code == 201

                second = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert second.status_code == 503
            assert second.json()["error"]["type"] == "session_limit_reached"
        finally:
            await pc.close()

    async def test_delete_location_hangs_up(self, server_env):
        """DELETE on the Location URL advertised by the 201 releases the unit;
        an unknown call id answers 404."""
        pc = RTCPeerConnection()
        try:
            pc.createDataChannel("oai-events")
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
                assert resp.status_code == 201
                location = resp.headers["location"]

                missing = await client.delete(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls/no-such-call",
                    timeout=10.0,
                )
                assert missing.status_code == 404

                hangup = await client.delete(f"http://127.0.0.1:{server_env.port}{location}", timeout=10.0)
                assert hangup.status_code == 200
        finally:
            await pc.close()
        await _wait_for_release(server_env)

    async def test_invalid_offer_releases_unit(self, server_env):
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                content="not an sdp",
                headers={"Content-Type": "application/sdp"},
                timeout=10.0,
            )
        assert resp.status_code == 400
        await _wait_for_release(server_env)

    async def test_setup_failure_releases_unit(self, server_env, monkeypatch):
        """A failure between claiming the unit and negotiate() (e.g. peer
        connection construction) must release the unit, not leak it."""
        import speech_to_speech.api.openai_realtime.websocket_router as router_module

        def _boom():
            raise RuntimeError("boom")

        # The calls endpoint uses the router's module-level binding (imported
        # eagerly at load), so that's the name to patch.
        monkeypatch.setattr(router_module, "rtc_configuration_from_env", _boom)

        pc = RTCPeerConnection()
        try:
            pc.createDataChannel("oai-events")
            pc.addTrack(AudioStreamTrack())
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://127.0.0.1:{server_env.port}/v1/realtime/calls",
                    content=pc.localDescription.sdp,
                    headers={"Content-Type": "application/sdp"},
                    timeout=10.0,
                )
            assert resp.status_code == 500
        finally:
            await pc.close()
        await _wait_for_release(server_env)


async def _wait_for_release(server_env, timeout: float = 5.0) -> None:
    """Assert the unit was released (or is draining) after a failed call."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = server_env.unit.session
        if session is None or session.released_at is not None:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("failed WebRTC call left the pipeline unit claimed")
