import sys
from threading import Event
from types import SimpleNamespace

from speech_to_speech.api.openai_realtime.local_client import LocalRealtimeAudioClient
from speech_to_speech.api.openai_realtime.server import RealtimeServer
from speech_to_speech.s2s_pipeline import build_pipeline, parse_arguments


def _default_args():
    original_argv = sys.argv[:]
    try:
        sys.argv = ["speech-to-speech"]
        return parse_arguments()
    finally:
        sys.argv = original_argv


def test_realtime_mode_builds_pipeline_unit_pool(monkeypatch):
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


def test_local_mode_adds_loopback_client_to_same_realtime_engine(monkeypatch):
    args = _default_args()
    args.module_kwargs.mode = "local"
    args.realtime_server_kwargs.host = "192.0.2.10"
    args.realtime_server_kwargs.port = 9876
    pipeline_handler = object()
    unit = SimpleNamespace(handlers=[pipeline_handler])
    monkeypatch.setattr("speech_to_speech.s2s_pipeline._build_pipeline_unit", lambda **_kwargs: unit)

    manager = build_pipeline(args, Event())

    assert manager.handlers[0] is pipeline_handler
    server = manager.handlers[1]
    client = manager.handlers[2]
    assert isinstance(server, RealtimeServer)
    assert isinstance(client, LocalRealtimeAudioClient)
    assert server.pool == [unit]
    assert server.host == "127.0.0.1"
    assert server.port == 9876
    assert client.config.host == "127.0.0.1"
    assert client.config.port == server.port
