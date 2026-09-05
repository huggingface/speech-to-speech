from queue import Queue
from threading import Event
from types import SimpleNamespace
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
    sid = service.register(routing=selection(stt={"model": "asr", "provider": "hf", "protocol": "transcriptions"}))
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
    handler = OpenAICompatibleSTTHandler(Event(), Queue(), Queue(), setup_kwargs={"warmup_enabled": False})
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
    cfg = RuntimeConfig(routing=selection(llm={"model": "text-llm", "provider": "hf", "protocol": protocol}))
    assert not cfg.accepts_audio_input


@pytest.mark.parametrize("audio_input", [False, True])
def test_llm_without_tts_returns_text_and_audio_input_uses_the_selected_model(audio_input):
    from speech_to_speech.LLM.chat import make_user_message
    from speech_to_speech.LLM.chat_completions_language_model import ChatCompletionsApiModelHandler
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
    from speech_to_speech.pipeline.messages import GenerateResponseRequest, TTSInput

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
    cfg.chat.add_item(make_user_message("Remember the blue bicycle."))
    handler = ChatCompletionsApiModelHandler(
        Event(),
        Queue(),
        Queue(),
        setup_kwargs={"api_key": "test", "stream": False, "warmup_enabled": False},
    )
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="It is blue.", tool_calls=[]))], usage=None
        )
    )
    handler.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    request = GenerateResponseRequest(runtime_config=cfg, audio=np.zeros(160) if audio_input else None)
    outputs = list(handler.process(request))
    assert create.call_args.kwargs["model"] == "audio-llm"
    messages = str(create.call_args.kwargs["messages"])
    assert "blue bicycle" in messages
    assert ("input_audio" in messages) is audio_input
    processor = LMOutputProcessor(Event(), Queue(), Queue())
    processed = [item for output in outputs for item in processor.process(output)]
    assert not any(isinstance(item, TTSInput) for item in processed)
