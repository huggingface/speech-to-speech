import json
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.testclient import TestClient

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClient
from speech_to_speech.api.openai_realtime.server import RealtimeServer
from speech_to_speech.s2s_pipeline import build_local_pipeline, build_pipeline, parse_arguments


@pytest.mark.parametrize("llm", ["chat-completions", "responses-api"])
def test_routed_pool_construction_does_not_probe_bootstrap_models(monkeypatch, llm):
    from speech_to_speech.api.openai_realtime.websocket_router import create_app
    from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
    from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler
    from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler

    args = parse_arguments(["--stt", "openai", "--llm-backend", llm, "--tts", "openai"], command="serve")
    args.module_kwargs.num_pipelines = 2
    args.realtime_server_kwargs.session_routing_enabled = True
    args.llm_backend.config.update(base_url="http://gateway/v1", api_key="test-key")
    monkeypatch.setattr("speech_to_speech.s2s_pipeline.VADHandler", lambda *a, **kw: SimpleNamespace())
    probes = []
    for handler in (OpenAICompatibleSTTHandler, BaseOpenAICompatibleHandler, OpenAICompatibleTTSHandler):
        probe = Mock(side_effect=RuntimeError("bootstrap model is unavailable"))
        monkeypatch.setattr(handler, "warmup", probe)
        probes.append(probe)
    manager = build_pipeline(args, Event())
    server = manager.handlers[-1]
    assert len(server.pool) == 2
    routes = {
        "id": "allocated-session",
        "pipeline": "healthy-alternative",
        "routes": {
            "stt": {"model": "healthy-stt", "provider": "test", "protocol": "transcriptions"},
            "llm": {
                "model": "healthy-llm",
                "provider": "test",
                "protocol": "chat_completions" if llm == "chat-completions" else "responses",
            },
            "tts": {"model": "healthy-tts", "provider": "test", "protocol": "speech", "voice": "alloy"},
        },
    }
    with TestClient(create_app(server.pool, server.stop_event, session_routing_enabled=True)) as client:
        with client.websocket_connect("/v1/realtime", headers={"X-Speech-Session-Routing": json.dumps(routes)}) as ws:
            event = ws.receive_json()
            assert event["type"] == "session.created"
            assert event["session"]["model"] == "healthy-llm"
    for probe in probes:
        probe.assert_not_called()
    # Per-unit routing setup must not mutate the configuration used by legacy builds.
    assert "warmup_enabled" not in args.stt_backend.config
    assert "warmup_enabled" not in args.llm_backend.config
    assert "warmup_enabled" not in args.tts_backend.config
    args.realtime_server_kwargs.session_routing_enabled = False
    with pytest.raises(RuntimeError, match="bootstrap model is unavailable"):
        build_pipeline(args, Event())


def _default_args():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        return parse_arguments()
    finally:
        sys.argv = original_argv


def test_serve_builds_pipeline_unit_pool(monkeypatch):
    args = _default_args()
    args.module_kwargs.num_pipelines = 2
    unit_handlers = [object(), object()]
    units = [SimpleNamespace(handlers=[handler]) for handler in unit_handlers]
    calls = []

    def fake_build_pipeline_unit(**kwargs):
        calls.append(kwargs)
        return units[kwargs["index"]]

    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", fake_build_pipeline_unit)
    stop_event = Event()
    manager = build_pipeline(args, stop_event)

    assert manager.handlers[:2] == unit_handlers
    assert len(manager.handlers) == 3
    assert isinstance(manager.handlers[-1], RealtimeServer)
    assert manager.handlers[-1].pool == units
    assert manager.handlers[-1].stop_event is stop_event
    assert [call["index"] for call in calls] == [0, 1]


def test_local_composes_loopback_client_with_same_server_builder(monkeypatch):
    args = _default_args()
    args.realtime_server_kwargs.host = "192.0.2.10"
    args.realtime_server_kwargs.port = 9876
    args.local_audio_kwargs.local_audio_playback_buffer_ms = 240
    pipeline_handler = object()
    unit = SimpleNamespace(handlers=[pipeline_handler])
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", lambda **_kwargs: unit)

    manager = build_local_pipeline(args, Event())

    assert manager.handlers[0] is pipeline_handler
    server = manager.handlers[1]
    client = manager.handlers[2]
    assert isinstance(server, RealtimeServer)
    assert isinstance(client, RealtimeAudioClient)
    assert server.pool == [unit]
    assert server.host == "127.0.0.1"
    assert server.port == 9876
    assert client.config.url == f"ws://127.0.0.1:{server.port}/v1/realtime"
    assert client.config.api_key == "local"
    assert client.config.playback_buffer_ms == 240


def test_local_resolves_backend_specific_playback_buffer_defaults(monkeypatch):
    unit = SimpleNamespace(handlers=[object()])
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", lambda **_kwargs: unit)

    cases = [
        (["--tts", "qwen3"], 0),
        (["--tts", "openai"], 196),
        (["--tts", "openai", "--playback-buffer-ms", "0"], 0),
        (["--tts", "openai", "--playback-buffer-ms", "240"], 240),
    ]
    for argv, expected_buffer_ms in cases:
        args = parse_arguments(argv, command="local")
        manager = build_local_pipeline(args, Event())
        client = manager.handlers[-1]

        assert isinstance(client, RealtimeAudioClient)
        assert client.config.playback_buffer_ms == expected_buffer_ms
