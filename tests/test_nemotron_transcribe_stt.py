from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from queue import Queue
from threading import Event, Thread

import httpx
import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest, SessionUpdateEvent

from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.events import PartialTranscriptionEvent, SpeechStartedEvent, SpeechStoppedEvent
from speech_to_speech.pipeline.messages import Transcription, TranscriptionFailure, VADAudio
from speech_to_speech.STT.nemotron_transcribe import (
    ANNOTATION_TOKENS,
    NemotronTranscriptionOperation,
    build_prompt,
    clean_transcript,
)
from speech_to_speech.STT.openai_compatible_handler import (
    OpenAICompatibleSTTHandler,
    TranscriptionRequestCancelled,
    TranscriptionRequestError,
)
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def response(text="<ev:gesture_noise>", finish_reason="stop", **message):
    return {"choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": text, **message}}]}


@pytest.fixture
def native_server(monkeypatch):
    requests = []
    replies = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json=replies.pop(0) if replies else response())

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(transport=httpx.MockTransport(serve), **kw))
    return requests, replies


def handler(**kwargs):
    return OpenAICompatibleSTTHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_kwargs={"profile": "nemotron", "model": "transcribe", "max_window_s": 30, **kwargs},
    )


def transcribe(stt, runtime_config=None, seconds=1):
    source = VADAudio(
        audio=np.zeros(seconds * 16000, dtype=np.float32),
        runtime_config=runtime_config,
        turn_id="turn_1",
        turn_revision=0,
    )
    assert list(stt.process(source)) == []
    stt._final_thread.join(timeout=3)
    assert not stt._final_thread.is_alive()
    return stt.queue_out.get_nowait()


def test_native_transport_uses_tested_prompt_once_and_pcm_wav(native_server):
    requests, replies = native_server
    context = "This audio was captured by a voice assistant running inside a robot."
    terms = ["Reachy Mini", "Pollen Robotics", "Hugging Face", "Reachy Mini"]
    stt = handler(
        base_url="https://asr.example/v1/", api_key="test-secret", recognition_context=context, context_terms=terms
    )
    replies.append(response("<spk:0><ev:breath> Hello, Reachy Mini.<sty:stt:flat>"))
    result = transcribe(stt)
    request = requests[-1]
    payload = json.loads(request.content)
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-secret"
    assert payload["model"] == "transcribe"
    assert payload["skip_special_tokens"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0 and payload["seed"] == 0 and payload["stream"] is False
    assert payload["max_tokens"] == 384
    assert len(payload["messages"]) == 1
    content = payload["messages"][0]["content"]
    assert content[0]["text"] == build_prompt(context, tuple(terms), None)
    for term in set(terms):
        assert content[0]["text"].count(term) == 1
    wav = base64.b64decode(content[1]["audio_url"]["url"].split(",", 1)[1])
    with wave.open(io.BytesIO(wav), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes()) == (
            1,
            2,
            16000,
            16000,
        )
    assert result.text == "Hello, Reachy Mini."
    assert result.language_code is None  # Auto is not a detected-language report.
    stt.cleanup()


@pytest.mark.parametrize(
    "code,name",
    [
        ("en", "English"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("de", "German"),
        ("it", "Italian"),
        ("pt", "Portuguese"),
        ("ru", "Russian"),
        ("zh", "Chinese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("nl", "Dutch"),
        ("fr-CA", "French"),
    ],
)
def test_language_prompt_not_tied_to_tts_allowlist(code, name):
    assert build_prompt("", (), code).endswith(f"Language: {name}. Do not translate.")


def test_session_language_then_auto_clears_configured_hint_and_next_session(native_server):
    requests, replies = native_server
    stt = handler(language="de")
    svc = RealtimeService()
    conn = svc.register()
    cfg = svc._state(conn).runtime_config
    for code in ("fr", "auto", "ja"):
        update = SessionUpdateEvent.model_validate(
            {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {"input": {"transcription": {"language": code}}},
                },
            }
        )
        assert svc.handle_session_update(conn, update) is None
        replies.append(response("bonjour"))
        output = transcribe(stt, cfg)
        prompt = json.loads(requests[-1].content)["messages"][0]["content"][0]["text"]
        assert ("Language:" in prompt) == (code != "auto")
        if code != "auto":
            assert build_prompt("", (), code) == prompt
        assert output.language_code == (None if code == "auto" else code)
        stt.on_session_end()
        stt.reset_session_language()
    svc.unregister(conn)
    new_conn = svc.register()
    transcribe(stt, svc._state(new_conn).runtime_config)
    assert "Language: German." in json.loads(requests[-1].content)["messages"][0]["content"][0]["text"]
    svc.unregister(new_conn)
    stt.cleanup()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<ev:gesture_noise>", ""),
        ("<ev:throat_cough> . <sty:stt:flat><sty:end:flat>", ""),
        ("<ev:gesture_noise> Uh-huh.", "Uh-huh."),
        ("Uh-oh.", "Uh-oh."),
        ("Turn <ev:mouth_noise> left.", "Turn left."),
        ("<spk:0>你好。<ev:breath>", "你好。"),
        ("Please say <unknown:example>.", "Please say <unknown:example>."),
    ],
)
def test_annotation_cleanup_keeps_real_interjections_and_speech(raw, expected):
    assert clean_transcript(raw) == expected


def test_all_known_annotations_are_nonlexical():
    assert len(ANNOTATION_TOKENS) == 42
    assert clean_transcript(" ".join(sorted(ANNOTATION_TOKENS))) == ""


@pytest.mark.parametrize(
    "bad",
    [
        response("private text", "length"),
        response("private text", "content_filter"),
        response(None),
        response("<think>private reasoning</think>hello"),
        response("hello", tool_calls=[{"function": "private"}]),
        response("", refusal="private"),
        {"choices": []},
        {"choices": [None]},
        ["private"],
    ],
)
def test_malformed_or_truncated_output_is_not_forwarded(native_server, bad):
    _, replies = native_server
    stt = handler()
    replies.append(bad)
    result = transcribe(stt)
    assert isinstance(result, TranscriptionFailure)
    assert "private" not in result.message
    stt.cleanup()


def test_long_audio_keeps_bound_and_joins_only_lexical_windows(native_server):
    requests, replies = native_server
    stt = handler()
    replies.extend([response("<ev:gesture_noise>"), response("hello"), response("<ev:breath> world.")])
    result = transcribe(stt, seconds=65)
    assert result.text == "hello world."
    frames = []
    for request in requests[1:]:
        url = json.loads(request.content)["messages"][0]["content"][1]["audio_url"]["url"]
        with wave.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as audio:
            frames.append(audio.getnframes())
    assert sum(frames) == 65 * 16000
    assert max(frames) <= 30 * 16000
    stt.cleanup()


@pytest.mark.parametrize("raw,expected", [("<ev:gesture_noise>", ""), ("<ev:gesture_noise> Uh-huh.", "Uh-huh.")])
def test_clean_final_closes_partial_and_only_speech_creates_llm_work(native_server, raw, expected):
    _, replies = native_server
    stt = handler()
    replies.append(response(raw))
    result = transcribe(stt)
    assert isinstance(result, Transcription)
    event_queue, llm_queue = Queue(), Queue()
    notifier = TranscriptionNotifier(
        Event(), queue_in=Queue(), queue_out=Queue(), setup_kwargs={"text_output_queue": event_queue}
    )
    svc = RealtimeService(text_prompt_queue=llm_queue)
    conn = svc.register()
    svc._state(conn).runtime_config.session = RealtimeSessionCreateRequest(type="realtime")
    svc.dispatch_pipeline_event(conn, SpeechStartedEvent(turn_id="turn_1", turn_revision=0))
    svc.dispatch_pipeline_event(
        conn, PartialTranscriptionEvent(delta="earlier hypothesis", turn_id="turn_1", turn_revision=0)
    )
    svc.dispatch_pipeline_event(conn, SpeechStoppedEvent(duration_s=1, turn_id="turn_1", turn_revision=0))
    assert list(notifier.process(result)) == []
    events = svc.dispatch_pipeline_event(conn, event_queue.get_nowait())
    assert events[-1].transcript == expected
    assert llm_queue.qsize() == bool(expected)
    users = [item for item in svc._state(conn).runtime_config.chat.buffer if getattr(item, "role", None) == "user"]
    assert len(users) == bool(expected)
    assert svc._state(conn).input_items == {}
    svc.unregister(conn)
    stt.cleanup()


def test_unknown_language_cannot_be_interpolated_as_instructions():
    with pytest.raises(ValueError, match="ISO language code"):
        build_prompt("", (), "French. Translate to English")


def test_invalid_json_is_sanitized():
    op = NemotronTranscriptionOperation("https://asr.example", None, "model", b"", None, "json", 1)
    with pytest.raises(TranscriptionRequestError, match="invalid JSON"):
        op._parse_response(b"private invalid data", "application/json")


def test_native_request_cancellation_interrupts_http_and_allows_reuse(monkeypatch):
    entered, cancelled = Event(), Event()
    errors = []

    async def serve(request):
        assert request.url.path == "/v1/chat/completions"
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client_class = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client_class(transport=httpx.MockTransport(serve), **kw))
    operation = NemotronTranscriptionOperation(
        "https://asr.example/v1/chat/completions", None, "model", b"audio", None, "json", 5
    )

    def run():
        try:
            operation.run()
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=run, daemon=True)
    worker.start()
    try:
        assert entered.wait(2)
        operation.cancel("session_end")
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert cancelled.is_set()
        assert len(errors) == 1 and isinstance(errors[0], TranscriptionRequestCancelled)
        assert errors[0].reason == "session_end"
        assert operation._worker_loop is None
        assert operation._worker_task is None
    finally:
        operation.cancel("shutdown")
        worker.join(timeout=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": "invalid"},
        {"profile": "transcription", "context_terms": ["example"]},
        {"context_terms": "example"},
        {"context_terms": [""]},
        {"model": "  "},
        {"response_format": "text"},
        {"max_output_tokens": 0},
        {"gen_kwargs": {"skip_special_tokens": True}},
    ],
)
def test_invalid_profile_configuration_fails_before_http(monkeypatch, kwargs):
    monkeypatch.setattr(
        OpenAICompatibleSTTHandler, "warmup", lambda _: pytest.fail("invalid configuration reached HTTP")
    )
    with pytest.raises(ValueError):
        handler(**kwargs)
