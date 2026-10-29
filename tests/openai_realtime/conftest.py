from queue import Queue
from threading import Event as ThreadingEvent

import pytest
from openai.types.realtime import RealtimeSessionCreateRequest
from openai.types.realtime.realtime_audio_config import RealtimeAudioConfig
from openai.types.realtime.realtime_audio_config_input import RealtimeAudioConfigInput
from openai.types.realtime.realtime_audio_config_output import RealtimeAudioConfigOutput
from openai.types.realtime.realtime_audio_formats import AudioPCM

from speech_to_speech.api.openai_realtime.pipeline_unit import NotifyingQueue, OutputSignal, PipelineUnit
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.cancel_scope import CancelScope


def make_pipeline_unit(*, index: int = 0, service: RealtimeService | None = None) -> PipelineUnit:
    """Build a unit whose output queues wake the event-driven send loop (#692)."""
    text_prompt_queue: Queue = Queue()
    should_listen = ThreadingEvent()
    should_listen.set()
    if service is None:
        service = RealtimeService(text_prompt_queue=text_prompt_queue, should_listen=should_listen)
    output_signal = OutputSignal()
    return PipelineUnit(
        index=index,
        service=service,
        cancel_scope=CancelScope(),
        should_listen=should_listen,
        response_playing=ThreadingEvent(),
        input_queue=Queue(),
        output_queue=NotifyingQueue(signal=output_signal),
        text_output_queue=NotifyingQueue(signal=output_signal),
        text_prompt_queue=text_prompt_queue,
        output_signal=output_signal,
        handlers=[],
    )


def _session_16k() -> RealtimeSessionCreateRequest:
    """Build a test session with 16 kHz audio rates (matches PIPELINE_SAMPLE_RATE)."""
    fmt = AudioPCM.model_construct(rate=16000, type="audio/pcm")
    return RealtimeSessionCreateRequest.model_construct(
        type="realtime",
        audio=RealtimeAudioConfig.model_construct(
            input=RealtimeAudioConfigInput.model_construct(format=fmt),
            output=RealtimeAudioConfigOutput.model_construct(format=fmt),
        ),
    )


@pytest.fixture
def runtime_config():
    cfg = RuntimeConfig()
    cfg.session = _session_16k()
    return cfg


@pytest.fixture
def text_prompt_queue():
    return Queue()


@pytest.fixture
def should_listen():
    ev = ThreadingEvent()
    ev.set()
    return ev


@pytest.fixture
def service(runtime_config, text_prompt_queue, should_listen):
    svc = RealtimeService(
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )
    return svc


@pytest.fixture
def conn_id(service, runtime_config):
    cid = service.register()
    service._state(cid).runtime_config = runtime_config
    yield cid
    service.unregister(cid)
