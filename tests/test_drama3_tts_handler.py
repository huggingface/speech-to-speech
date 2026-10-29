import base64
import json
from queue import Queue
from threading import Event

import httpx
import numpy as np
import pytest
from scipy.io import wavfile

from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.TTS.drama3_handler import Drama3TTSHandler
from speech_to_speech.TTS.openai_compatible_handler import SpeechRequestError


def _handler(**kwargs):
    return Drama3TTSHandler(Event(), queue_in=Queue(), queue_out=Queue(), setup_args=(Event(),), setup_kwargs=kwargs)


@pytest.mark.parametrize("clone", [False, True])
def test_drama3_sglang_request_and_resampled_audio(monkeypatch, tmp_path, clone):
    requests = []
    samples = np.round(10000 * np.sin(2 * np.pi * 440 * np.arange(4410) / 44100)).astype("<i2")
    pcm = samples.tobytes()

    class AudioStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for offset in range(0, len(pcm), 997):
                yield pcm[offset : offset + 997]

    def respond(request):
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "audio/pcm"}, stream=AudioStream())

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.delenv("DRAMA3_API_KEY", raising=False)
    reference = tmp_path / "voice.wav"
    wavfile.write(reference, 44100, samples)
    kwargs = {"ref_audio": str(reference), "ref_text": "Reference words."} if clone else {}
    handler = _handler(base_url="https://speech.example/v1/", **kwargs)
    assert len(requests) == 1  # Startup validates the real request shape too.
    chunks = list(handler.process(TTSInput(text="[laughs] Hello!", selected_language="fr", tts_language_code="fr")))
    assert handler.queue_out.empty()
    assert all(chunk.shape == (512,) and chunk.dtype == np.int16 for chunk in chunks)
    audio = np.concatenate(chunks)
    assert len(audio) == 2048
    assert np.any(audio[:1600])
    assert np.all(audio[1600:] == 0)
    request = requests[-1]
    assert str(request.url) == "https://speech.example/v1/audio/speech"
    assert "Authorization" not in request.headers
    assert json.loads(request.content) == {
        "model": "fishaudio/drama-3",
        "input": "[laughs] Hello!",
        "references": [{"data": base64.b64encode(reference.read_bytes()).decode(), "text": "Reference words."}]
        if clone
        else [],
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": 4097,
    }


def test_drama3_credentials_and_sampling(monkeypatch):
    monkeypatch.setattr(Drama3TTSHandler, "warmup", lambda self: None)
    monkeypatch.setenv("DRAMA3_API_KEY", "env-token")
    assert _handler().api_key == "env-token"
    handler = _handler(api_key="explicit-token", gen_kwargs={"temperature": 0.7, "top_p": 0.8})
    assert handler.api_key == "explicit-token"
    payload = handler._request_payload(text="Hello", voice="alloy")
    assert payload["temperature"] == 0.7 and payload["top_p"] == 0.8
    assert _handler(api_key="").api_key == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ref_audio": "voice.wav"},
        {"ref_text": "Reference words."},
        {"ref_audio": "voice.wav", "ref_text": " "},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
    ],
)
def test_invalid_configuration_fails_before_warmup(monkeypatch, kwargs):
    def unexpected_warmup(self):
        pytest.fail("Invalid configuration reached the endpoint")

    monkeypatch.setattr(Drama3TTSHandler, "warmup", unexpected_warmup)
    with pytest.raises(ValueError):
        _handler(**kwargs)


def test_drama3_startup_rejects_incompatible_server(monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(422, text="unsupported speech fields"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    with pytest.raises(SpeechRequestError, match="HTTP 422"):
        _handler()
