from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.LLM.utils import language_name_for_prompt
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.s2s_pipeline import _stt_session_languages, parse_arguments
from speech_to_speech.STT.nemo_asr_handler import FARSI_MODEL, NemoASRSTTHandler
from tests.test_nemo_asr_handler import _install_fake_nemo


@pytest.fixture
def farsi_model(monkeypatch):
    calls = []

    class Encoder:
        streaming_cfg = SimpleNamespace(drop_extra_pre_encoded=7)

        def set_default_att_context_size(self, context):
            calls.append(("context", context))

        def get_initial_cache_state(self, batch_size):
            assert batch_size == 1
            return ["channel", "time", "length"]

    class Model:
        encoder = Encoder()

        @classmethod
        def restore_from(cls, path, map_location):
            calls.append(("restore", path, map_location))
            return cls()

        @classmethod
        def from_pretrained(cls, **kwargs):
            pytest.fail("Persian checkpoint must be restored from its .nemo file")

        def to(self, device=None, **kwargs):
            calls.append(("device", device, kwargs))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

        def change_decoding_strategy(self, config):
            calls.append(("decoding", config))

        def set_inference_prompt(self, prompt):
            calls.append(("prompt", prompt))

        def transcribe(self, *args, **kwargs):
            pytest.fail("transcribe() randomizes the language prompt")

        def conformer_stream_step(self, **kwargs):
            calls.append(("step", kwargs))
            if kwargs["previous_hypotheses"] is None:
                assert kwargs["previous_pred_out"] is None
                assert kwargs["cache_last_channel"] == "channel"
                assert kwargs["drop_extra_pre_encoded"] == 0
                return "prediction", [SimpleNamespace(text="سلام")], "c1", "t1", "l1", "hypothesis"
            assert kwargs["previous_hypotheses"] == "hypothesis"
            assert kwargs["previous_pred_out"] == "prediction"
            assert kwargs["cache_last_channel"] == "c1"
            assert kwargs["cache_last_time"] == "t1"
            assert kwargs["cache_last_channel_len"] == "l1"
            assert kwargs["drop_extra_pre_encoded"] == 7
            assert kwargs["keep_all_outputs"] is True
            return "prediction2", [SimpleNamespace(text="سلام می‌خواهم کتاب بخوانم")], "c2", "t2", "l2", "hypothesis2"

    class Buffer:
        def __init__(self, model, online_normalization, pad_and_drop_preencoded):
            assert online_normalization is False
            assert pad_and_drop_preencoded is False
            self.step = 0

        def append_audio(self, audio, stream_id):
            assert audio.dtype == np.float32
            assert stream_id == -1

        def __iter__(self):
            for self.step in (0, 1):
                yield np.zeros(10), np.array([10])

        def is_buffer_empty(self):
            return self.step == 1

    _install_fake_nemo(monkeypatch, Model)
    for name in (
        "nemo.collections.asr.parts.submodules.rnnt_decoding",
        "nemo.collections.asr.parts.utils.streaming_utils",
    ):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["nemo.collections.asr.parts.submodules.rnnt_decoding"].RNNTDecodingConfig = lambda **kwargs: kwargs
    sys.modules["nemo.collections.asr.parts.utils.streaming_utils"].CacheAwareStreamingAudioBuffer = Buffer
    omega = types.ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(structured=lambda config: config)
    monkeypatch.setitem(sys.modules, "omegaconf", omega)

    def download(repo_id, filename):
        calls.append(("download", repo_id, filename))
        return "/tmp/farsi.nemo"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    monkeypatch.setattr("speech_to_speech.STT.nemo_asr_handler.console.print", lambda *args, **kwargs: None)
    return Model, calls


@pytest.mark.parametrize("language", ["en", "auto", "fa", "fa-IR"])
def test_farsi_setup_restores_checkpoint_and_forces_trained_prompt(farsi_model, language):
    _, calls = farsi_model
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(model_name=FARSI_MODEL, device="cpu", language=language)

    assert ("download", FARSI_MODEL, "nemotron-asr-streaming-farsi.nemo") in calls
    assert ("restore", "/tmp/farsi.nemo", "cpu") in calls
    assert ("context", [56, 13]) in calls
    assert ("decoding", {"fused_batch_size": -1}) in calls
    assert ("eval",) in calls
    assert handler.language == handler.start_language == handler.last_language == "fa"
    assert all(call[1] == "fa-IR" for call in calls if call[0] == "prompt")


def test_farsi_final_progressive_and_repeated_turns_flush_and_reset_caches(farsi_model):
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(model_name=FARSI_MODEL, device="cpu")
    for mode in ("progressive", "final", "final"):
        events = list(
            handler.process(
                VADAudio(
                    audio=np.zeros(16000, dtype=np.float64),
                    mode=mode,
                    turn_id="turn_fa",
                    turn_revision=2,
                    speech_end_at_s=123.0,
                )
            )
        )
        assert len(events) == 1
        event = events[0]
        assert event.text == "سلام می‌خواهم کتاب بخوانم"
        assert event.turn_id == "turn_fa"
        assert event.turn_revision == 2
        if mode == "progressive":
            assert isinstance(event, PartialTranscription)
        else:
            assert isinstance(event, Transcription)
            assert event.language_code == "fa"
            assert event.speech_stopped_at_s == 123.0
            assert language_name_for_prompt(event.language_code, enable=True) == "persian"
    handler.on_session_end()
    assert handler.last_language == "fa"


def test_farsi_empty_audio_does_not_decode(farsi_model):
    _, calls = farsi_model
    handler = object.__new__(NemoASRSTTHandler)
    handler.setup(model_name=FARSI_MODEL, device="cpu")
    calls.clear()
    assert handler._transcribe(np.array([], dtype=np.float32)) == ""
    assert calls == []


def test_farsi_warmup_failure_prevents_startup(farsi_model, monkeypatch):
    def fail(self, audio):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(NemoASRSTTHandler, "_transcribe_farsi", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        object.__new__(NemoASRSTTHandler).setup(model_name=FARSI_MODEL, device="cpu")


def test_farsi_cli_and_session_language_validation():
    args = parse_arguments(["--stt", "nemotron-streaming", "--nemotron_streaming_model_name", FARSI_MODEL])
    assert args.stt_backend.config["model_name"] == FARSI_MODEL
    assert args.stt_backend.spec.required_extra == "nemo"
    assert _stt_session_languages(args.stt_backend, SimpleNamespace(_is_farsi=True)) == {"fa"}
