import pytest
from queue import Queue
from threading import Event as ThreadingEvent

from api.openai_realtime.service import RealtimeService
from api.openai_realtime.runtime_config import RuntimeConfig


@pytest.fixture
def runtime_config():
    cfg = RuntimeConfig()
    cfg.client_audio_rate = 16000
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
    return RealtimeService(
        runtime_config=runtime_config,
        text_prompt_queue=text_prompt_queue,
        should_listen=should_listen,
    )


@pytest.fixture
def conn_id(service):
    cid = "test-conn-1"
    service.register(cid)
    yield cid
    service.unregister(cid)
