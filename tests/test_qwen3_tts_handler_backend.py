from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import TTS.qwen3_tts_handler as qwen3_tts_module
from pipeline_messages import MessageTag
from TTS.qwen3_tts_handler import Qwen3TTSHandler


def test_setup_uses_mlx_backend_on_darwin_and_maps_qwen_repo_ids(monkeypatch):
    recorded = {}

    def _setup_mlx(self, model_name):
        recorded["model_name"] = model_name

    def _setup_faster(self, *args, **kwargs):
        raise AssertionError("Darwin setup should not use the faster-qwen3-tts backend")

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device="cuda",
    )

    assert handler.backend == "mlx"
    assert handler.device == "mps"
    assert handler.dtype is None
    assert handler.streaming_chunk_size == 4
    assert (
        recorded["model_name"]
        == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"
    )


@pytest.mark.parametrize("quantization", ["4bit", "6bit", "8bit"])
def test_setup_supports_quantized_mlx_mapping_on_darwin(monkeypatch, quantization):
    recorded = {}

    def _setup_mlx(self, model_name):
        recorded["model_name"] = model_name

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mlx_quantization=quantization,
    )

    assert handler.backend == "mlx"
    assert handler.mlx_quantization == quantization
    assert (
        recorded["model_name"]
        == f"mlx-community/Qwen3-TTS-12Hz-0.6B-Base-{quantization}"
    )


def test_setup_preserves_explicit_mlx_model_suffix_when_quantization_unset(monkeypatch):
    recorded = {}

    def _setup_mlx(self, model_name):
        recorded["model_name"] = model_name

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    )

    assert handler.backend == "mlx"
    assert handler.mlx_quantization is None
    assert recorded["model_name"] == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"


def test_setup_preserves_faster_backend_off_darwin(monkeypatch):
    recorded = {}

    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("Non-Darwin setup should not use the mlx backend")

    def _setup_faster(self, model_name, dtype, attn_implementation):
        recorded["model_name"] = model_name
        recorded["dtype"] = dtype
        recorded["attn_implementation"] = attn_implementation

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda",
        dtype="float16",
        attn_implementation="sdpa",
    )

    assert handler.backend == "faster_qwen3_tts"
    assert handler.streaming_chunk_size == 8
    assert recorded == {
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    }


def test_setup_preserves_explicit_chunk_size_on_darwin(monkeypatch):
    def _setup_mlx(self, model_name):
        return None

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(
        Event(),
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        streaming_chunk_size=4,
    )

    assert handler.backend == "mlx"
    assert handler.streaming_chunk_size == 4


def test_setup_warns_when_non_streaming_mode_set_on_darwin(monkeypatch, caplog):
    def _setup_mlx(self, model_name):
        return None

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)

    with caplog.at_level("WARNING"):
        handler.setup(
            Event(),
            model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            non_streaming_mode=True,
        )

    assert "mlx-audio does not expose non_streaming_mode yet" in caplog.text


def test_setup_rejects_invalid_mlx_quantization(monkeypatch):
    def _setup_mlx(self, model_name):
        return None

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)

    with pytest.raises(ValueError, match="Unsupported qwen3_tts_mlx_quantization"):
        handler.setup(
            Event(),
            model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            mlx_quantization="5bit",
        )


def test_mlx_helper_methods_use_model_config_and_streaming_conversion():
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.model_name = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    handler.speaker = None
    handler.streaming_chunk_size = 8
    handler.model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type="custom_voice"),
        get_supported_speakers=lambda: ["Vivian", "Ryan"],
    )

    assert handler._model_type() == "custom_voice"
    assert handler._resolve_speaker() == "Vivian"
    assert handler._mlx_streaming_interval() == pytest.approx(0.64)


@pytest.mark.parametrize("override", [True, False])
def test_install_faster_non_streaming_mode_override_patches_custom_prepare(override):
    recorded = {}
    fake_model = SimpleNamespace(
        _warmed_up=True,
        model=SimpleNamespace(
            _build_assistant_text=lambda text: f"assistant:{text}",
            _tokenize_texts=lambda texts: [texts[0]],
            _build_instruct_text=lambda text: f"instruct:{text}",
            model=SimpleNamespace(
                talker=SimpleNamespace(rope_deltas=None),
                config=SimpleNamespace(talker_config=SimpleNamespace()),
            ),
        ),
        _prepare_generation_custom=lambda *args, **kwargs: None,
        _build_talker_inputs_local=lambda **kwargs: (
            recorded.update(kwargs),
            ("tie", "tam", "tth", "tpe"),
        )[1],
    )

    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "faster_qwen3_tts"
    handler.non_streaming_mode = override
    handler.model = fake_model

    handler._install_faster_non_streaming_mode_override()
    handler.model._prepare_generation_custom(
        text="Hello there.",
        language="English",
        speaker="Vivian",
        instruct="calm",
    )

    assert recorded["non_streaming_mode"] is override
    assert recorded["languages"] == ["English"]
    assert recorded["speakers"] == ["Vivian"]


def test_prepare_mlx_ref_audio_normalizes_file_and_caches_result(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    save_calls = []

    fake_sf = SimpleNamespace(
        read=lambda path, always_2d=False, dtype=None: (
            [[0.1, 0.2], [0.3, 0.4]],
            44100,
        ),
        write=lambda path, waveform, sample_rate, format=None, subtype=None: (
            save_calls.append((path, sample_rate, format, subtype)),
            Path(path).write_bytes(b"RIFF"),
        ),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "mlx"
    handler.model = SimpleNamespace(sample_rate=24000)
    handler._mlx_ref_audio_cache = {}
    handler._mlx_temp_ref_audio_files = set()

    normalized = handler._prepare_mlx_ref_audio(str(source))
    normalized_again = handler._prepare_mlx_ref_audio(str(source))

    assert normalized == normalized_again
    assert Path(normalized).exists()
    assert save_calls == [(normalized, 24000, "WAV", "PCM_16")]


def test_apply_session_voice_override_warns_for_non_file_for_base_model(caplog):
    handler = object.__new__(Qwen3TTSHandler)
    handler.runtime_config = SimpleNamespace(
        session=SimpleNamespace(
            audio=SimpleNamespace(
                output=SimpleNamespace(voice="alloy")
            )
        )
    )
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None

    with caplog.at_level("WARNING"):
        handler._apply_session_voice_override("base")

    assert handler.ref_audio == "TTS/ref_audio.wav"
    assert handler.speaker is None
    assert "Ignoring Qwen3-TTS session voice override" in caplog.text


def test_process_only_reenables_listening_after_end_of_response(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.runtime_config = None
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type: None
    handler._process_voice_clone = lambda text: iter([np.zeros(512, dtype=np.int16)])

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process("Hello there."))

    assert len(outputs) == 1
    assert handler.should_listen.is_set() is False

    end_outputs = list(handler.process((MessageTag.END_OF_RESPONSE, None)))

    assert end_outputs == [qwen3_tts_module.AUDIO_RESPONSE_DONE]
    assert handler.should_listen.is_set() is True


def test_process_reenables_listening_when_generation_fails_outside_realtime(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.runtime_config = None
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type: None

    def _boom(text):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    handler._process_voice_clone = _boom

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process("Hello there."))

    assert outputs == []
    assert handler.should_listen.is_set() is True


def test_process_voice_clone_passes_non_streaming_mode_to_faster_backend(monkeypatch):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.runtime_config = None
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.ref_text = "Reference text."
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.xvec_only = False
    handler.parity_mode = False
    handler.non_streaming_mode = False
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 360
    handler.blocksize = 512
    handler.backend = "faster_qwen3_tts"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(tts_model_type="base")),
        generate_voice_clone_streaming=lambda **kwargs: (
            captured.update(kwargs),
            iter([(np.zeros(512, dtype=np.float32), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process("Hello there."))

    assert len(outputs) == 1
    assert captured["non_streaming_mode"] is False


def test_process_voice_clone_omits_non_streaming_mode_when_unset(monkeypatch):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.runtime_config = None
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.ref_text = "Reference text."
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.xvec_only = False
    handler.parity_mode = False
    handler.non_streaming_mode = None
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 360
    handler.blocksize = 512
    handler.backend = "faster_qwen3_tts"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(tts_model_type="base")),
        generate_voice_clone_streaming=lambda **kwargs: (
            captured.update(kwargs),
            iter([(np.zeros(512, dtype=np.float32), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process("Hello there."))

    assert len(outputs) == 1
    assert "non_streaming_mode" not in captured
