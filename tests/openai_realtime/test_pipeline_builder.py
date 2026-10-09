import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClient
from speech_to_speech.api.openai_realtime.server import RealtimeServer
from speech_to_speech.s2s_pipeline import build_local_pipeline, build_pipeline, parse_arguments


@pytest.fixture(autouse=True)
def mock_shared_client(monkeypatch):
    # These tests replace every model handler; no provider is contacted.
    monkeypatch.setattr("openai.OpenAI", MagicMock())


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


def test_failed_handler_setup_cleans_and_preserves_original_error():
    from queue import Queue

    import pytest

    from speech_to_speech.baseHandler import BaseHandler

    cleaned = []
    original = RuntimeError("original setup error")

    class FailingHandler(BaseHandler):
        def setup(self):
            raise original

        def cleanup(self):
            cleaned.append(True)
            raise RuntimeError("cleanup error")

    with pytest.raises(RuntimeError) as exc:
        FailingHandler(Event(), Queue(), Queue())
    assert exc.value is original
    assert cleaned == [True]


def test_pipeline_ledger_records_handlers_before_later_factory_failure(monkeypatch):
    import pytest

    from speech_to_speech import s2s_pipeline

    args = _default_args()
    ledger = []
    vad = SimpleNamespace()
    monkeypatch.setattr(s2s_pipeline, "VADHandler", lambda *args, **kwargs: vad)
    monkeypatch.setattr(
        s2s_pipeline, "create_backend_handler", lambda *args: (_ for _ in ()).throw(ValueError("failure"))
    )
    with pytest.raises(ValueError):
        s2s_pipeline._build_pipeline_unit(
            index=0,
            stop_event=Event(),
            module_kwargs=args.module_kwargs,
            vad_handler_kwargs=args.vad_handler_kwargs,
            stt_backend=args.stt_backend,
            llm_backend=args.llm_backend,
            tts_backend=args.tts_backend,
            resource_ledger=ledger,
        )
    assert ledger == [vad]


def test_remote_handler_closes_owned_client_once():
    from speech_to_speech.LLM.base_openai_compatible_language_model import BaseOpenAICompatibleHandler
    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

    closed = []
    handler = object.__new__(ResponsesApiModelHandler)
    handler.client = SimpleNamespace(close=lambda: closed.append(True))
    BaseOpenAICompatibleHandler.cleanup(handler)
    BaseOpenAICompatibleHandler.cleanup(handler)
    assert closed == [True]


def test_remote_warmup_failure_closes_allocated_client_and_keeps_error(monkeypatch):
    from queue import Queue

    import pytest

    from speech_to_speech.LLM.responses_api_language_model import ResponsesApiModelHandler

    original = RuntimeError("warmup failure")
    closed = []
    monkeypatch.setattr(
        "speech_to_speech.LLM.base_openai_compatible_language_model.OpenAI",
        lambda **kwargs: SimpleNamespace(close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(ResponsesApiModelHandler, "warmup", lambda self: (_ for _ in ()).throw(original))
    with pytest.raises(RuntimeError) as exc:
        ResponsesApiModelHandler(Event(), Queue(), Queue(), setup_kwargs={"api_key": "test"})
    assert exc.value is original
    assert closed == [True]


def test_diarizer_ledger_registration_precedes_warmup(monkeypatch):
    import pytest

    from speech_to_speech import s2s_pipeline
    from speech_to_speech.diarization import StreamingDiarizer

    args = _default_args()
    args.module_kwargs.diarization_model_name = "test-model"
    ledger = []
    vad = SimpleNamespace()
    diarizer = SimpleNamespace(
        sample_rate=args.vad_handler_kwargs.sample_rate, warmup=lambda: (_ for _ in ()).throw(ValueError("warmup"))
    )
    monkeypatch.setattr(s2s_pipeline, "VADHandler", lambda *args, **kwargs: vad)
    monkeypatch.setattr(s2s_pipeline, "resolve_device", lambda *args: "cpu")
    monkeypatch.setattr(StreamingDiarizer, "from_pretrained", lambda *args, **kwargs: diarizer)
    with pytest.raises(ValueError):
        s2s_pipeline._build_pipeline_unit(
            index=0,
            stop_event=Event(),
            module_kwargs=args.module_kwargs,
            vad_handler_kwargs=args.vad_handler_kwargs,
            stt_backend=args.stt_backend,
            llm_backend=args.llm_backend,
            tts_backend=args.tts_backend,
            resource_ledger=ledger,
        )
    assert ledger == [vad, diarizer]
