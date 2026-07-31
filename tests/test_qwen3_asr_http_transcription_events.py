import numpy as np

from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.STT import qwen3_asr_http_handler
from speech_to_speech.STT.qwen3_asr_http_handler import (
    Qwen3ASRHTTPSTTHandler,
    _encode_wav_data_uri,
    _parse_asr_output,
)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeHTTPClient:
    def __init__(self, content: str = "language English<asr_text>hello there") -> None:
        self.content = content
        self.last_json = None

    def post(self, url, json):
        self.last_json = json
        return _FakeResponse(self.content)


def _handler(content: str = "language English<asr_text>hello there"):
    handler = object.__new__(Qwen3ASRHTTPSTTHandler)
    handler.base_url = "http://127.0.0.1:8000"
    handler.client = _FakeHTTPClient(content)
    return handler


def test_parse_asr_output_with_tag():
    language, text = _parse_asr_output("language Chinese<asr_text>你好")
    assert language == "Chinese"
    assert text == "你好"


def test_parse_asr_output_without_tag_is_plain_text():
    language, text = _parse_asr_output("just plain text, no tag")
    assert language == ""
    assert text == "just plain text, no tag"


def test_parse_asr_output_empty_audio():
    language, text = _parse_asr_output("language None<asr_text>")
    assert language == ""
    assert text == ""


def test_encode_wav_data_uri_roundtrip_header():
    audio = np.zeros(1600, dtype=np.float32)
    uri = _encode_wav_data_uri(audio, sample_rate=16000)
    assert uri.startswith("data:audio/wav;base64,")


def test_qwen3_asr_http_transcription_is_final(monkeypatch):
    monkeypatch.setattr(qwen3_asr_http_handler.console, "print", lambda *args, **kwargs: None)

    handler = _handler()
    result = list(
        handler.process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode="final",
                turn_id="turn_1",
                turn_revision=2,
                created_at_s=123.0,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], Transcription)
    assert result[0].text == "hello there"
    assert result[0].language_code == "en"
    assert result[0].turn_id == "turn_1"
    assert result[0].turn_revision == 2
    assert result[0].speech_stopped_at_s == 123.0

    sent = handler.client.last_json
    assert sent["messages"][0]["content"][0]["type"] == "audio_url"
    assert sent["messages"][0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
