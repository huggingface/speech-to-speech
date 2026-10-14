from dataclasses import dataclass
from queue import Queue
from threading import Event

import pytest
import torch

import speech_to_speech.s2s_pipeline as s2s_pipeline
from speech_to_speech import configured_runtime as runtime
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.backend_registry import BackendSelection, BackendSpec
from speech_to_speech.config import ConfigurationError, load_config, resolve_config
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.s2s_pipeline import parse_arguments, prepare_all_args


@dataclass
class FakeArguments:
    fake_option: str = "default"


@pytest.fixture(autouse=True)
def restore_torch_threads():
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def _build_handlers(monkeypatch, module_kwargs):
    """Build a handler chain whose VAD behaves like Silero and record the thread count the STT sees."""
    stt_threads = []

    class SileroLikeVAD:
        def __init__(self, *_args, **_kwargs):
            # Loading the real silero_vad module calls torch.set_num_threads(1).
            torch.set_num_threads(1)

    class DummyHandler:
        def __init__(self, *_args, **_kwargs):
            pass

    def stt_factory(_context, _config):
        stt_threads.append(torch.get_num_threads())
        return object()

    def other_factory(_context, _config):
        return object()

    stt_spec = BackendSpec("cpu-stt", "stt", FakeArguments, stt_factory)
    llm_spec = BackendSpec("fake-llm", "llm", FakeArguments, other_factory)
    tts_spec = BackendSpec("fake-tts", "tts", FakeArguments, other_factory)
    monkeypatch.setattr(s2s_pipeline, "VADHandler", SileroLikeVAD)
    monkeypatch.setattr(s2s_pipeline, "TranscriptionNotifier", DummyHandler)
    monkeypatch.setattr("speech_to_speech.LLM.lm_output_processor.LMOutputProcessor", DummyHandler)

    s2s_pipeline._build_handlers(
        stop_event=Event(),
        should_listen=Event(),
        recv_audio_chunks_queue=Queue(),
        spoken_prompt_queue=Queue(),
        stt_output_queue=Queue(),
        text_prompt_queue=Queue(),
        lm_response_queue=Queue(),
        lm_processed_queue=Queue(),
        send_audio_chunks_queue=Queue(),
        text_output_queue=Queue(),
        module_kwargs=module_kwargs,
        vad_handler_kwargs=VADHandlerArguments(),
        stt_backend=BackendSelection(stt_spec, stt_spec.normalize(FakeArguments())),
        llm_backend=BackendSelection(llm_spec, llm_spec.normalize(FakeArguments())),
        tts_backend=BackendSelection(tts_spec, tts_spec.normalize(FakeArguments())),
        speculative_turns=SpeculativeTurnTracker(),
        cancel_scope=CancelScope(),
        pipeline_index=0,
        response_playing=Event(),
    )
    return stt_threads


def test_torch_num_threads_is_unset_by_default():
    assert ModuleArguments().torch_num_threads is None
    assert parse_arguments([]).module_kwargs.torch_num_threads is None
    assert parse_arguments(["--torch_num_threads", "6"]).module_kwargs.torch_num_threads == 6


def test_default_keeps_silero_single_thread_setting(monkeypatch):
    torch.set_num_threads(4)

    stt_threads = _build_handlers(monkeypatch, ModuleArguments())

    assert stt_threads == [1]
    assert torch.get_num_threads() == 1


def test_explicit_value_survives_silero_for_every_pipeline(monkeypatch):
    module_kwargs = ModuleArguments(torch_num_threads=3)

    # Each pipeline instance loads its own VAD; the setting must hold for all of them.
    first = _build_handlers(monkeypatch, module_kwargs)
    second = _build_handlers(monkeypatch, module_kwargs)

    assert first == [3]
    assert second == [3]
    assert torch.get_num_threads() == 3


@pytest.mark.parametrize("value", ["0", "-2"])
def test_cli_rejects_non_positive_values(value):
    args = parse_arguments(["--torch_num_threads", value])

    with pytest.raises(ValueError, match="--torch_num_threads must be a positive integer"):
        prepare_all_args(args)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "4"])
def test_validation_rejects_values_that_are_not_positive_integers(value):
    with pytest.raises(ValueError, match="--torch_num_threads must be a positive integer"):
        s2s_pipeline.validate_torch_num_threads(value)


CONFIG = """schema_version: 1
runtime:
  torch_num_threads: {value}
blocks:
  vad: {{kind: vad, backend: silero}}
  stt: {{kind: stt, backend: openai}}
  llm: {{kind: llm, backend: chat-completions}}
  tts: {{kind: tts, backend: openai}}
pipelines:
  primary:
    num_pipelines: 2
    stages: {{vad: vad, stt: stt, llm: llm, tts: tts}}
  secondary:
    stages: {{vad: vad, stt: stt, llm: llm, tts: tts}}
"""


def _resolve(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG.format(value=value))
    return resolve_config(load_config(path), include_client=True, environ={})


def test_yaml_runtime_setting_applies_to_every_pipeline_definition(tmp_path):
    adapted = runtime._adapt(_resolve(tmp_path, 4))

    assert {name: args.module_kwargs.torch_num_threads for name, args in adapted.items()} == {
        "primary": 4,
        "secondary": 4,
    }


def test_yaml_runtime_setting_rejects_non_positive_values(tmp_path):
    with pytest.raises(ConfigurationError):
        runtime._adapt(_resolve(tmp_path, 0))
