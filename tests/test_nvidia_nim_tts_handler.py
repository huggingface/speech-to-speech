"""NVIDIA Speech NIM provider protocol and pipeline integration."""

from __future__ import annotations

import io
import logging
import re
import wave
from queue import Queue
from threading import Event

import httpx
import numpy as np
import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.backend_registry import TTS_BACKENDS, HandlerContext, create_backend_handler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import ResponseFailedEvent
from speech_to_speech.pipeline.messages import TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments, prepare_all_args
from speech_to_speech.TTS.nvidia_nim_tts_handler import (
    NvidiaNimTTSHandler,
    SpeechRequestError,
    fix_date_commas,
    split_text,
)


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def _pcm_body(samples, fragment=None):
    pcm = np.asarray(samples, dtype="<i2").tobytes()
    if fragment is None:
        return pcm
    return _Chunks([pcm[i : i + fragment] for i in range(0, len(pcm), fragment)])


def _wav_body(samples, sample_rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


def _transport(monkeypatch, respond):
    requests = []
    original = httpx.AsyncClient

    def handle(request):
        request.read()
        requests.append(request)
        return respond(request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
    )
    return requests


def _audio_transport(monkeypatch, body_for=lambda request: _pcm_body(np.zeros(512)), content_type=None):
    def respond(request):
        body = body_for(request)
        headers = {"Content-Type": content_type} if content_type else {}
        if isinstance(body, httpx.AsyncByteStream):
            return httpx.Response(200, headers=headers, stream=body, request=request)
        return httpx.Response(200, headers=headers, content=body, request=request)

    return _transport(monkeypatch, respond)


def _form(request):
    return dict(
        re.findall(rb'Content-Disposition: form-data; name="([^"]+)"\r\n\r\n(.*?)\r\n--', request.content, re.DOTALL)
    )


def _handler(**kwargs):
    return NvidiaNimTTSHandler(
        Event(), Queue(), Queue(), setup_args=(Event(),), setup_kwargs={"sample_rate": 16000, **kwargs}
    )


@pytest.mark.parametrize("fragment", [1, 3, 37, 1023])
@pytest.mark.parametrize("sample_count", [100, 512, 1500])
def test_streaming_pcm_across_odd_chunk_boundaries(monkeypatch, fragment, sample_count):
    samples = np.arange(sample_count, dtype="<i2") - 700
    requests = _audio_transport(monkeypatch, lambda request: _pcm_body(samples, fragment), "application/octet-stream")
    chunks = list(_handler().process(TTSInput(text="Hello")))
    assert str(requests[-1].url) == "http://localhost:9000/v1/audio/synthesize_online"
    assert all(chunk.dtype == np.int16 and chunk.shape == (512,) for chunk in chunks)
    output = np.concatenate(chunks)
    assert len(output) == ((sample_count + 511) // 512) * 512
    np.testing.assert_array_equal(output[:sample_count], samples)
    assert np.all(output[sample_count:] == 0)


def test_offline_wav_is_parsed(monkeypatch):
    samples = np.arange(1000, dtype="<i2")
    requests = _audio_transport(monkeypatch, lambda request: _wav_body(samples), "audio/wav")
    chunks = list(_handler(stream=False, base_url="http://nim:9000/").process(TTSInput(text="Hello")))
    assert str(requests[-1].url) == "http://nim:9000/v1/audio/synthesize"
    output = np.concatenate(chunks)
    np.testing.assert_array_equal(output[:1000], samples)
    assert np.all(output[1000:] == 0)


def test_offline_wav_is_resampled_to_pipeline_rate(monkeypatch):
    _audio_transport(monkeypatch, lambda request: _wav_body(np.zeros(22050), sample_rate=22050))
    chunks = list(_handler(stream=False, sample_rate=22050).process(TTSInput(text="Hello")))
    assert sum(chunk.size for chunk in chunks) == ((16000 + 511) // 512) * 512


def test_form_fields_and_no_auth_header_without_key(monkeypatch):
    requests = _audio_transport(monkeypatch)
    handler = _handler(voice="Magpie-Multilingual.EN-US.Aria.Happy", language="en-US", sample_rate=22050)
    assert len(requests) == 1  # Warm-up is one short synthesis.
    assert list(handler.process(TTSInput(text="Hello there.")))
    request = requests[-1]
    assert request.headers["Content-Type"].startswith("multipart/form-data")
    assert "Authorization" not in request.headers
    assert _form(request) == {
        b"text": b"Hello there.",
        b"language": b"en-US",
        b"voice": b"Magpie-Multilingual.EN-US.Aria.Happy",
        b"sample_rate_hz": b"22050",
    }


def test_bearer_header_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    requests = _audio_transport(monkeypatch)
    _handler(api_key="nvapi-test")
    assert requests[0].headers["Authorization"] == "Bearer nvapi-test"


def test_openai_key_is_not_forwarded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    requests = _audio_transport(monkeypatch)
    _handler()
    assert "Authorization" not in requests[0].headers


def _voice_inputs():
    return [
        TTSInput(text="Hello", response=RealtimeResponseCreateParams(audio={"output": {"voice": "Aiden"}})),
        TTSInput(
            text="Hello",
            runtime_config=RuntimeConfig(
                session=RealtimeSessionCreateRequest(
                    type="realtime", audio={"output": {"voice": "Magpie-Multilingual.EN-US.Mia.Calm"}}
                )
            ),
        ),
        TTSInput(text="Hello"),
    ]


def test_client_voice_used_only_when_nim_style(monkeypatch, caplog):
    requests = _audio_transport(monkeypatch)
    handler = _handler(voice="Magpie-Multilingual.EN-US.Aria")
    with caplog.at_level(logging.INFO, logger="speech_to_speech.TTS.nvidia_nim_tts_handler"):
        for tts_input in _voice_inputs() + _voice_inputs():
            assert list(handler.process(tts_input))
    assert [_form(request)[b"voice"] for request in requests[1:]] == [
        b"Magpie-Multilingual.EN-US.Aria",
        b"Magpie-Multilingual.EN-US.Mia.Calm",
        b"Magpie-Multilingual.EN-US.Aria",
    ] * 2
    assert len([r for r in caplog.records if "ignoring client voice" in r.message]) == 1
    handler.on_session_end()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="speech_to_speech.TTS.nvidia_nim_tts_handler"):
        list(handler.process(_voice_inputs()[0]))
    assert len([r for r in caplog.records if "ignoring client voice" in r.message]) == 1


def test_voice_override_ignores_client_voices(monkeypatch):
    requests = _audio_transport(monkeypatch)
    handler = _handler(voice_override=" Magpie-Multilingual.EN-US.Aria.Angry ")
    for tts_input in _voice_inputs():
        assert list(handler.process(tts_input))
    assert {_form(request)[b"voice"] for request in requests} == {b"Magpie-Multilingual.EN-US.Aria.Angry"}
    assert handler.voice == "Magpie-Multilingual.EN-US.Aria"


@pytest.mark.parametrize("voice_override", ["", "   "])
def test_blank_voice_override_is_unset(monkeypatch, voice_override):
    requests = _audio_transport(monkeypatch)
    handler = _handler(voice="Magpie-Multilingual.EN-US.Aria", voice_override=voice_override)
    assert handler.voice_override is None
    for tts_input in _voice_inputs():
        assert list(handler.process(tts_input))
    assert [_form(request)[b"voice"] for request in requests] == [
        b"Magpie-Multilingual.EN-US.Aria",
        b"Magpie-Multilingual.EN-US.Aria",
        b"Magpie-Multilingual.EN-US.Mia.Calm",
        b"Magpie-Multilingual.EN-US.Aria",
    ]


def test_long_text_is_split_on_sentences_in_order(monkeypatch):
    requests = _audio_transport(monkeypatch)
    handler = _handler()
    sentences = [f"Sentence number {i} is here and it keeps going for a while." for i in range(60)]
    text = " ".join(sentences)
    assert len(text) > 2000
    assert list(handler.process(TTSInput(text=text)))
    texts = [_form(request)[b"text"].decode() for request in requests[1:]]
    assert len(texts) >= 2
    assert all(len(piece) <= 1800 for piece in texts)
    assert all(piece.endswith(".") for piece in texts)
    assert " ".join(texts) == text


def test_text_at_limit_is_sent_whole(monkeypatch):
    requests = _audio_transport(monkeypatch)
    text = "a. " * 599 + "abc"
    assert len(text) == 1800
    list(_handler().process(TTSInput(text=text)))
    assert len(requests) == 2


def _sentences(length):
    text = " ".join(f"Sentence number {i} is here and it keeps going for a while." for i in range(100))
    return text[: text.rfind(".", 0, length) + 1]


@pytest.mark.parametrize(
    ("voice", "length", "limit"),
    [("Magpie-Multilingual.EN-US.Aria", 1900, 1800), ("Chatterbox-Multilingual.EN-US.Female-1", 600, 450)],
)
def test_model_aware_limit_splits_before_dispatch(monkeypatch, voice, length, limit):
    requests = _audio_transport(monkeypatch)
    text = _sentences(length)
    assert limit < len(text) < 2000
    handler = _handler(voice=voice)
    assert handler.max_chars == limit
    assert list(handler.process(TTSInput(text=text)))
    texts = [_form(request)[b"text"].decode() for request in requests[1:]]
    assert len(texts) == 2
    assert all(len(piece) <= limit for piece in texts)
    assert " ".join(texts) == text


def test_explicit_max_chars_overrides_auto(monkeypatch):
    requests = _audio_transport(monkeypatch)
    handler = _handler(voice="Chatterbox-Multilingual.EN-US.Female-1", max_chars=1000)
    assert list(handler.process(TTSInput(text=_sentences(600))))
    assert len(requests) == 2


def test_cancel_during_first_piece_stops_remaining_pieces(monkeypatch):
    scope = CancelScope()
    requests = _audio_transport(monkeypatch)
    handler = _handler(cancel_scope=scope)
    chunks = handler.process(TTSInput(text=_sentences(4000)))
    next(chunks)
    assert len(requests) == 2
    scope.cancel()
    assert list(chunks) == []
    assert len(requests) == 2


def test_cjk_text_splits_at_full_stop():
    sentence = "这是一个很长的中文句子。"
    text = sentence * 200
    pieces = split_text(text)
    assert len(pieces) >= 2
    assert all(len(piece) <= 1800 and piece.endswith("。") for piece in pieces)
    assert "".join(pieces) == text


def test_split_text_handles_oversized_sentences():
    text = ("word " * 500).strip() + ". " + "x" * 4000
    pieces = split_text(text)
    assert all(len(piece) <= 1800 for piece in pieces)
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "Magpie-Zeroshot.EN-US.Speaker"},
        {"voice_override": "magpie-tts-zeroshot.EN-US.Speaker"},
    ],
)
def test_zeroshot_voice_fails_at_setup(monkeypatch, kwargs):
    requests = _audio_transport(monkeypatch)
    with pytest.raises(ValueError, match="Magpie Zeroshot needs an audio prompt"):
        _handler(**kwargs)
    assert requests == []


def test_short_ssml_is_sent_untouched(monkeypatch):
    requests = _audio_transport(monkeypatch)
    text = '<speak>Hello <break time="300ms"/> <phoneme alphabet="ipa" ph="wɝld">world</phoneme>.</speak>'
    assert list(_handler().process(TTSInput(text=text)))
    assert len(requests) == 2
    assert _form(requests[-1])[b"text"].decode() == text


@pytest.mark.parametrize(("sentences", "requests_expected"), [(40, 1), (80, 2)])
def test_long_ssml_is_split_into_valid_speak_documents(monkeypatch, caplog, sentences, requests_expected):
    requests = _audio_transport(monkeypatch)
    body = " ".join(
        f'Sentence {i} has a <break time="200ms"/> pause and <say-as interpret-as="characters">abc</say-as> &amp; more.'
        for i in range(sentences)
    )
    text = f'<speak version="1.0">{body}</speak>'
    assert len(text) > 1800
    with caplog.at_level(logging.WARNING, logger="speech_to_speech.TTS.nvidia_nim_tts_handler"):
        assert list(_handler().process(TTSInput(text=text)))
    texts = [_form(request)[b"text"].decode() for request in requests[1:]]
    assert len(texts) == requests_expected
    assert all(len(piece) <= 1800 for piece in texts)
    assert all(re.fullmatch(r"<speak>[^<>]+</speak>", piece) for piece in texts)
    assert all(piece[len("<speak>")].strip() and piece[-len("</speak>") - 1] == "." for piece in texts)
    assert "Sentence 0 has a pause and abc &amp; more." in texts[0]
    assert any("pronunciation tags were removed" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("November 4, 2026", "November 4 2026"),
        ("October 29, 2026", "October 29 2026"),
        ("November 4th, 2026", "November 4th 2026"),
        ("November 4 2026", "November 4 2026"),
        ("Nov. 4th, 2026", "Nov. 4th 2026"),
        ("Wednesday, November 4, 2026", "Wednesday, November 4 2026"),
        ("Dec 1, 1999", "Dec 1 1999"),
        ("We meet on Jan. 2nd, 2030, at noon.", "We meet on Jan. 2nd 2030, at noon."),
    ],
)
def test_fix_date_commas_drops_day_year_comma(text, expected):
    assert fix_date_commas(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Hello, world", "4, 5, 6", "In 2026, we shipped it.", "Route 66, 2026 edition", "November 4, 2150"],
)
def test_fix_date_commas_leaves_other_commas(text):
    assert fix_date_commas(text) == text


@pytest.mark.parametrize(("enabled", "sent"), [(True, b"Today is November 4 2026."), (False, None)])
def test_date_comma_fix_in_form_field(monkeypatch, enabled, sent):
    text = "Today is November 4, 2026."
    requests = _audio_transport(monkeypatch)
    assert list(_handler(fix_date_commas=enabled).process(TTSInput(text=text)))
    assert _form(requests[-1])[b"text"] == (sent or text.encode())


def test_http_400_body_is_surfaced(monkeypatch):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, content=_pcm_body(np.zeros(512)), request=request)
        return httpx.Response(
            400,
            headers={"Content-Type": "application/json"},
            json={"error": "Voice Magpie-Multilingual.EN-US.Nobody not found"},
            request=request,
        )

    _transport(monkeypatch, respond)
    handler = _handler()
    assert list(handler.process(TTSInput(text="Hello", response_key="response-1"))) == []
    failure = handler.queue_out.get_nowait()
    assert isinstance(failure, ResponseFailedEvent)
    assert failure.response_key == "response-1"
    assert "HTTP 400" in failure.message
    assert "Voice Magpie-Multilingual.EN-US.Nobody not found" in failure.message


def test_warmup_failure_surfaces_body(monkeypatch):
    _transport(monkeypatch, lambda request: httpx.Response(503, text="model not ready", request=request))
    with pytest.raises(SpeechRequestError, match="HTTP 503: model not ready"):
        _handler()


@pytest.mark.parametrize("kwargs", [{"sample_rate": 0}, {"timeout_s": 0}, {"timeout_s": float("nan")}])
def test_invalid_settings_fail_before_network_dispatch(monkeypatch, kwargs):
    requests = _audio_transport(monkeypatch)
    with pytest.raises(ValueError):
        _handler(**kwargs)
    assert requests == []


def test_cli_registry_builds_handler(monkeypatch):
    requests = _audio_transport(monkeypatch)
    monkeypatch.setattr(
        "sys.argv",
        [
            "speech-to-speech",
            "--tts",
            "nvidia_nim",
            "--nvidia_nim_tts_base_url",
            "http://localhost:9000",
            "--nvidia_nim_tts_voice",
            "Magpie-Multilingual.EN-US.Aria.Happy",
        ],
    )
    args = parse_arguments()
    prepare_all_args(args)
    context = HandlerContext(
        stop_event=Event(),
        queue_in=Queue(),
        queue_out=Queue(),
        text_output_queue=Queue(),
        should_listen=Event(),
        cancel_scope=CancelScope(),
        speculative_turns=SpeculativeTurnTracker(),
        pipeline_index=0,
        sample_rate=16000,
        enable_live_transcription=False,
        live_transcription_update_interval=0.5,
    )
    assert args.tts_backend.spec is TTS_BACKENDS["nvidia_nim"]
    handler = create_backend_handler(args.tts_backend, context)
    assert isinstance(handler, NvidiaNimTTSHandler)
    assert handler.voice == "Magpie-Multilingual.EN-US.Aria.Happy"
    assert (handler.language, handler.sample_rate, handler.stream) == ("en-US", 22050, True)
    assert handler.fix_date_commas is True
    assert handler.max_chars == 1800
    assert handler.cancel_scope is context.cancel_scope
    assert handler.speculative_turns is context.speculative_turns
    assert str(requests[0].url) == "http://localhost:9000/v1/audio/synthesize_online"
