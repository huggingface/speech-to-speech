from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread

import numpy as np

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.STT.openai_compatible_handler import HttpTranscriptionOperation
from speech_to_speech.TTS import openai_compatible_handler as tts_module
from speech_to_speech.TTS.openai_compatible_handler import OpenAICompatibleTTSHandler


class _TranscriptionServer(BaseHTTPRequestHandler):
    received_path = ""
    received_body = b""

    def do_POST(self) -> None:
        type(self).received_path = self.path
        length = int(self.headers["content-length"])
        type(self).received_body = self.rfile.read(length)
        body = json.dumps({"text": "hello", "language": "en"}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


def test_http_transcription_operation_uploads_wav_multipart():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model="test-model",
            wav_bytes=b"RIFF-test-wave",
            language="en",
            response_format="json",
            timeout_s=2,
        )
        result = operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert result.text == "hello"
    assert result.language == "en"
    assert _TranscriptionServer.received_path == "/v1/audio/transcriptions"
    assert b'form-data; name="model"' in _TranscriptionServer.received_body
    assert b"test-model" in _TranscriptionServer.received_body
    assert b'filename="audio.wav"' in _TranscriptionServer.received_body
    assert b"RIFF-test-wave" in _TranscriptionServer.received_body


def test_http_transcription_operation_can_select_nim_model_by_language():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TranscriptionServer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        operation = HttpTranscriptionOperation(
            endpoint_url=f"http://127.0.0.1:{server.server_port}/v1/audio/transcriptions",
            api_key=None,
            model=None,
            wav_bytes=b"RIFF-test-wave",
            language="en-US",
            response_format="json",
            timeout_s=2,
        )
        operation.run()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert b'form-data; name="model"' not in _TranscriptionServer.received_body
    assert b'form-data; name="language"' in _TranscriptionServer.received_body
    assert b"en-US" in _TranscriptionServer.received_body


class _FakeSpeechOperation:
    instances: list["_FakeSpeechOperation"] = []

    def __init__(self, **kwargs) -> None:
        self.payload = kwargs["payload"]
        self.cancelled = False
        type(self).instances.append(self)

    def iter_bytes(self, cancel_check):
        samples = np.arange(2400, dtype="<i2")
        encoded = samples.tobytes()
        for offset in range(0, len(encoded), 301):
            if cancel_check():
                return
            yield encoded[offset : offset + 301]

    def cancel(self) -> None:
        self.cancelled = True


def test_openai_tts_streams_resampled_fixed_size_pcm(monkeypatch):
    _FakeSpeechOperation.instances.clear()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice": "aiden",
            "language": "Auto",
            "sample_rate": 24000,
            "blocksize": 512,
            "cancel_scope": CancelScope(),
        },
    )

    chunks = list(handler.process(TTSInput(text="Hello", language_code="en")))

    assert chunks
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    assert sum(chunk.size for chunk in chunks) == 2048
    payload = _FakeSpeechOperation.instances[0].payload
    assert payload["input"] == "Hello"
    assert payload["voice"] == "aiden"
    assert payload["language"] == "English"
    assert payload["stream"] is True
    assert payload["stream_format"] == "audio"


def test_openai_tts_cancellation_stops_publication_and_closes_operation(monkeypatch):
    _FakeSpeechOperation.instances.clear()
    monkeypatch.setattr(tts_module, "HttpSpeechOperation", _FakeSpeechOperation)
    cancel_scope = CancelScope()
    handler = OpenAICompatibleTTSHandler(
        Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        setup_args=(Event(),),
        setup_kwargs={
            "sample_rate": 24000,
            "blocksize": 512,
            "cancel_scope": cancel_scope,
        },
    )

    generation = handler.process(TTSInput(text="Hello", cancel_generation=cancel_scope.generation))
    first = next(generation)
    cancel_scope.cancel()

    assert isinstance(first, np.ndarray)
    assert list(generation) == []
    assert _FakeSpeechOperation.instances[0].cancelled is True
