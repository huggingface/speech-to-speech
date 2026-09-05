from queue import Queue
from threading import Event
from unittest.mock import Mock

import numpy as np
import pytest
from openai.types.realtime import ResponseCreateEvent

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.api.openai_realtime.session_routing import SessionRouting
from speech_to_speech.pipeline.events import AudioInputCompletedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import VADAudio
from speech_to_speech.STT.openai_compatible_handler import OpenAICompatibleSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier


def selection(*, stt=None, llm=None, tts=None):
    return SessionRouting.model_validate(
        {
            "id": "allocated-session",
            "pipeline": "empty",
            "updates_enabled": True,
            "routes": {"stt": stt, "llm": llm, "tts": tts},
        }
    )


def test_empty_session_exposes_disabled_stages_and_rejects_generation():
    queue = Queue()
    service = RealtimeService(text_prompt_queue=queue, should_listen=Event())
    sid = service.register(routing=selection())
    created = service.build_session_created(sid).model_dump()
    assert created["session"]["model"] is None
    assert created["session"]["models"] == {"stt": None, "llm": None, "tts": None}
    assert created["session"]["output_modalities"] == ["text"]
    error = service.handle_response_create(sid, ResponseCreateEvent(type="response.create"))
    assert error.type == "error"
    assert "LLM" in error.error.message
    assert queue.empty()


def test_transcription_only_session_records_text_without_generating():
    queue = Queue()
    service = RealtimeService(text_prompt_queue=queue, should_listen=Event())
    sid = service.register(
        routing=selection(stt={"model": "asr", "provider": "hf", "protocol": "transcriptions"})
    )
    events = service.dispatch_pipeline_event(sid, TranscriptionCompletedEvent(transcript="Keep this transcript."))
    assert any(e.type == "conversation.item.input_audio_transcription.completed" for e in events)
    assert "Keep this transcript." in str(service._state(sid).runtime_config.chat.to_transformers_chat())
    assert queue.empty()


def test_disabling_stt_for_audio_llm_bypasses_transcription_and_preserves_audio():
    cfg = RuntimeConfig(
        routing=selection(
            llm={
                "model": "audio-llm",
                "provider": "hf",
                "protocol": "chat_completions",
                "capabilities": {"audio_input": True, "context_window": 32768},
            }
        )
    )
    handler = OpenAICompatibleSTTHandler(
        Event(), Queue(), Queue(), setup_kwargs={"warmup_enabled": False}
    )
    handler._make_operation = Mock(side_effect=AssertionError("disabled STT must not call a model"))
    audio = VADAudio(audio=np.arange(160, dtype=np.float32), runtime_config=cfg, mode="final")
    assert list(handler.process(audio)) == [audio]
    progressive = audio.model_copy(update={"mode": "progressive"})
    assert list(handler.process(progressive)) == []
    events = Queue()
    notifier = TranscriptionNotifier(Event(), Queue(), Queue(), setup_kwargs={"text_output_queue": events})
    assert list(notifier.process(audio)) == []
    event = events.get_nowait()
    assert isinstance(event, AudioInputCompletedEvent)
    np.testing.assert_array_equal(event.audio, audio.audio)
    assert event.audio_sample_rate == 16000


@pytest.mark.parametrize("protocol", ["responses", "chat_completions"])
def test_audio_bypass_requires_an_explicit_compatible_audio_route(protocol):
    cfg = RuntimeConfig(
        routing=selection(llm={"model": "text-llm", "provider": "hf", "protocol": protocol})
    )
    assert not cfg.accepts_audio_input
