import logging
import sys
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

import speech_to_speech.TTS.qwen3_tts_handler as qwen3_tts_module
from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler


def _audible_stream_chunk():
    return np.full(512, 0.1, dtype=np.float32)


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
    assert recorded["model_name"] == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit"


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
    assert recorded["model_name"] == f"mlx-community/Qwen3-TTS-12Hz-0.6B-Base-{quantization}"


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

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        recorded["model_name"] = model_name
        recorded["dtype"] = dtype
        recorded["attn_implementation"] = attn_implementation
        recorded["backend"] = backend

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
    assert handler.faster_backend == "ggml"
    assert handler.streaming_chunk_size == 8
    assert recorded == {
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "dtype": "float16",
        "attn_implementation": "sdpa",
        "backend": "ggml",
    }


def test_setup_passes_torch_backend_override_off_darwin(monkeypatch):
    recorded = {}

    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("Non-Darwin setup should not use the mlx backend")

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        recorded["backend"] = backend

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event(), backend="torch")

    assert handler.backend == "faster_qwen3_tts"
    assert handler.faster_backend == "torch"
    assert recorded["backend"] == "torch"


def test_setup_defaults_to_custom_voice_profile_off_darwin(monkeypatch):
    recorded = {}

    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("Non-Darwin setup should not use the mlx backend")

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        recorded["model_name"] = model_name
        recorded["dtype"] = dtype
        recorded["attn_implementation"] = attn_implementation
        recorded["backend"] = backend

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event())

    assert handler.backend == "faster_qwen3_tts"
    assert handler.faster_backend == "ggml"
    assert recorded["model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert recorded["backend"] == "ggml"
    assert handler.ref_audio is None
    assert handler.speaker == "Aiden"
    assert handler.language == "auto"
    assert handler.non_streaming_mode is True


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("zh", "chinese"),
        ("zh-CN", "chinese"),
        ("zh_Hans", "chinese"),
        ("Chinese", "chinese"),
        ("en-US", "english"),
        ("English", "english"),
        ("Auto", "auto"),
        ("", "auto"),
    ],
)
def test_setup_normalizes_qwen3_language_aliases(monkeypatch, language, expected):
    def _setup_mlx(self, *args, **kwargs):
        raise AssertionError("Non-Darwin setup should not use the mlx backend")

    def _setup_faster(self, model_name, dtype, attn_implementation, backend):
        return None

    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_faster", _setup_faster)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)
    handler.setup(Event(), language=language)

    assert handler.language == expected


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


def test_setup_logs_when_non_streaming_mode_set_on_darwin(monkeypatch, caplog):
    def _setup_mlx(self, model_name):
        return None

    monkeypatch.setattr(qwen3_tts_module, "platform", "darwin")
    monkeypatch.setattr(Qwen3TTSHandler, "_setup_mlx", _setup_mlx)
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)

    with caplog.at_level("DEBUG"):
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


def test_setup_rejects_invalid_faster_backend(monkeypatch):
    monkeypatch.setattr(qwen3_tts_module, "platform", "linux")
    monkeypatch.setattr(Qwen3TTSHandler, "warmup", lambda self: None)

    handler = object.__new__(Qwen3TTSHandler)

    with pytest.raises(ValueError, match="Unsupported qwen3_tts_backend"):
        handler.setup(Event(), backend="cuda")


@pytest.mark.parametrize("faster_backend", ["ggml", "torch"])
def test_warmup_uses_public_faster_backend_api(faster_backend):
    calls = []
    generated = []
    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "faster_qwen3_tts"
    handler.faster_backend = faster_backend
    handler.parity_mode = False
    handler.model = SimpleNamespace(
        warmup=lambda **kwargs: calls.append(kwargs),
    )
    handler._warmup_process = lambda text: generated.append(text) or iter(())

    handler.warmup()

    assert calls == [{"prefill_len": 100}]
    assert generated == ["Hello, this is a warmup."]


def test_warmup_logs_backend_neutral_failure(caplog):
    def fail_warmup(**_kwargs):
        raise RuntimeError("boom")

    handler = object.__new__(Qwen3TTSHandler)
    handler.backend = "faster_qwen3_tts"
    handler.faster_backend = "ggml"
    handler.parity_mode = False
    handler.model = SimpleNamespace(warmup=fail_warmup)
    handler._warmup_process = lambda _text: iter(())

    with caplog.at_level(logging.WARNING):
        handler.warmup()

    assert "Qwen3-TTS backend warmup failed: boom" in caplog.text
    assert "CUDA graph capture failed" not in caplog.text


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
    fake_cfg = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="alloy"))))
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None

    with caplog.at_level("WARNING"):
        handler._apply_session_voice_override("base", runtime_config=fake_cfg)

    assert handler.ref_audio == "TTS/ref_audio.wav"
    assert handler.speaker is None
    assert "Ignoring Qwen3-TTS session voice override" in caplog.text


def test_apply_session_voice_override_ignores_unsupported_custom_voice_speaker(caplog):
    handler = object.__new__(Qwen3TTSHandler)
    fake_cfg = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="cedar"))))
    handler.ref_audio = None
    handler.speaker = "Aiden"
    handler.model = SimpleNamespace(model=SimpleNamespace(get_supported_speakers=lambda: ["aiden", "vivian"]))

    with caplog.at_level("WARNING"):
        handler._apply_session_voice_override("custom_voice", runtime_config=fake_cfg)

    assert handler.ref_audio is None
    assert handler.speaker == "Aiden"
    assert "not a supported CustomVoice speaker" in caplog.text
    assert "cedar" in caplog.text


def test_apply_session_voice_override_accepts_supported_custom_voice_speaker():
    handler = object.__new__(Qwen3TTSHandler)
    fake_cfg = SimpleNamespace(session=SimpleNamespace(audio=SimpleNamespace(output=SimpleNamespace(voice="Vivian"))))
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = "Aiden"
    handler.model = SimpleNamespace(model=SimpleNamespace(get_supported_speakers=lambda: ["aiden", "vivian"]))

    handler._apply_session_voice_override("custom_voice", runtime_config=fake_cfg)

    assert handler.ref_audio is None
    assert handler.speaker == "vivian"


def test_process_only_reenables_listening_after_end_of_response(monkeypatch):
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type, runtime_config=None, response=None: None
    handler._process_voice_clone = lambda text: iter([np.zeros(512, dtype=np.int16)])

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.", runtime_config=RuntimeConfig())))

    assert len(outputs) == 1
    assert handler.should_listen.is_set() is False

    end_outputs = list(handler.process(EndOfResponse()))

    assert end_outputs == [AUDIO_RESPONSE_DONE]


def test_process_waits_for_pending_reopen_and_drops_stale_tts_input():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    handler = object.__new__(Qwen3TTSHandler)
    handler.speculative_turns = tracker
    done = Event()
    outputs = []

    def run_process():
        outputs.extend(
            handler.process(
                TTSInput(
                    text="stale",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_process)
    thread.start()

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []


def test_process_waits_for_pending_reopen_and_drops_stale_end_of_response():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    handler = object.__new__(Qwen3TTSHandler)
    handler.speculative_turns = tracker
    done = Event()
    outputs = []

    def run_process():
        outputs.extend(handler.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
        done.set()

    thread = Thread(target=run_process)
    thread.start()

    assert not done.wait(0.05)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []


def test_process_waits_for_reopen_grace_and_drops_stale_tts_input():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.5)
    handler = object.__new__(Qwen3TTSHandler)
    handler.speculative_turns = tracker
    done = Event()
    outputs = []

    def run_process():
        outputs.extend(
            handler.process(
                TTSInput(
                    text="stale",
                    turn_id="turn_1",
                    turn_revision=0,
                )
            )
        )
        done.set()

    thread = Thread(target=run_process)
    thread.start()

    assert not done.wait(0.05)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []


def test_process_waits_for_reopen_grace_and_drops_stale_end_of_response():
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    tracker.start_reopen_grace("turn_1", 0, grace_s=0.5)
    handler = object.__new__(Qwen3TTSHandler)
    handler.speculative_turns = tracker
    done = Event()
    outputs = []

    def run_process():
        outputs.extend(handler.process(EndOfResponse(turn_id="turn_1", turn_revision=0)))
        done.set()

    thread = Thread(target=run_process)
    thread.start()

    assert not done.wait(0.05)
    candidate_revision = tracker.begin_reopen_candidate("turn_1", 0)
    assert tracker.confirm_reopen_candidate("turn_1", 0, candidate_revision)
    assert done.wait(1.0)
    thread.join(timeout=1.0)

    assert outputs == []


def test_process_commits_turn_before_generating_audio(monkeypatch, caplog):
    tracker = SpeculativeTurnTracker()
    tracker.observe("turn_1", 0)
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.speculative_turns = tracker
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type, runtime_config=None, response=None: None

    def _process_voice_clone(text):
        assert tracker.is_committed("turn_1", 0)
        yield np.zeros(512, dtype=np.int16)

    handler._process_voice_clone = _process_voice_clone

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    with caplog.at_level(logging.INFO, logger="speech_to_speech.TTS.qwen3_tts_handler"):
        outputs = list(
            handler.process(
                TTSInput(
                    text="Hello there.",
                    turn_id="turn_1",
                    turn_revision=0,
                    speech_stopped_at_s=qwen3_tts_module.perf_counter() - 1.0,
                )
            )
        )

    assert len(outputs) == 1
    assert tracker.is_committed("turn_1", 0)
    assert "Last speech detected to first speech out:" in caplog.text


def test_process_does_not_set_should_listen_when_generation_fails(monkeypatch):
    """TTS no longer manages should_listen; the I/O streamer does via AUDIO_RESPONSE_DONE."""
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.backend = "mlx"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(config=SimpleNamespace(tts_model_type="base"))
    handler._apply_session_voice_override = lambda model_type, runtime_config=None, response=None: None

    def _boom(text):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    handler._process_voice_clone = _boom

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.")))

    assert outputs == []
    assert handler.should_listen.is_set() is False


def test_process_voice_clone_passes_non_streaming_mode_to_faster_backend(monkeypatch):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
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
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.")))

    assert len(outputs) == 1
    assert captured["non_streaming_mode"] is False


def test_process_voice_clone_passes_none_non_streaming_mode_when_unset(monkeypatch):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
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
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.")))

    assert len(outputs) == 1
    assert captured["non_streaming_mode"] is None


@pytest.mark.parametrize("override", [None, False, True])
def test_process_custom_voice_passes_non_streaming_mode_to_faster_backend(monkeypatch, override):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = None
    handler.ref_text = "Reference text."
    handler.speaker = "Vivian"
    handler.instruct = "calm"
    handler.language = "English"
    handler.xvec_only = False
    handler.parity_mode = False
    handler.non_streaming_mode = override
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 360
    handler.blocksize = 512
    handler.backend = "faster_qwen3_tts"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(tts_model_type="custom_voice")),
        generate_custom_voice_streaming=lambda **kwargs: (
            captured.update(kwargs),
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.")))

    assert len(outputs) == 1
    assert captured["non_streaming_mode"] is override


@pytest.mark.parametrize("override", [None, False, True])
def test_process_voice_design_passes_non_streaming_mode_to_faster_backend(monkeypatch, override):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = None
    handler.ref_text = "Reference text."
    handler.speaker = None
    handler.instruct = "bright radio voice"
    handler.language = "English"
    handler.xvec_only = False
    handler.parity_mode = False
    handler.non_streaming_mode = override
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 360
    handler.blocksize = 512
    handler.backend = "faster_qwen3_tts"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(tts_model_type="voice_design")),
        generate_voice_design_streaming=lambda **kwargs: (
            captured.update(kwargs),
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    outputs = list(handler.process(TTSInput(text="Hello there.")))

    assert len(outputs) == 1
    assert captured["non_streaming_mode"] is override


def test_estimate_max_new_tokens_scales_with_utterance_length():
    handler = object.__new__(Qwen3TTSHandler)
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 1536

    short_budget = handler._estimate_max_new_tokens("Hello there.")
    long_text = " ".join(["This is a deliberately long sentence for the Qwen3 TTS budget estimator."] * 12)
    long_budget = handler._estimate_max_new_tokens(long_text)

    assert short_budget == 360
    assert long_budget > short_budget
    assert long_budget % handler.streaming_chunk_size == 0
    assert long_budget <= handler.max_new_tokens


def test_estimate_max_new_tokens_respects_configured_cap():
    handler = object.__new__(Qwen3TTSHandler)
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 400

    long_text = " ".join(["This is a deliberately long sentence for the Qwen3 TTS budget estimator."] * 12)

    assert handler._estimate_max_new_tokens(long_text) == 400


def test_estimate_max_new_tokens_can_exceed_default_ceiling_when_raised():
    handler = object.__new__(Qwen3TTSHandler)
    handler.streaming_chunk_size = 8
    handler.max_new_tokens = 2400

    long_text = " ".join(["This is a deliberately long sentence for the Qwen3 TTS budget estimator."] * 30)

    assert handler._estimate_max_new_tokens(long_text) > 1536


def test_process_voice_clone_scales_max_new_tokens_for_faster_backend(monkeypatch):
    captured = {}
    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
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
    handler.max_new_tokens = 1536
    handler.blocksize = 512
    handler.backend = "faster_qwen3_tts"
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(tts_model_type="base")),
        generate_voice_clone_streaming=lambda **kwargs: (
            captured.update(kwargs),
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)

    long_text = " ".join(["This is a deliberately long sentence for the faster Qwen3 TTS backend."] * 12)
    outputs = list(handler.process(TTSInput(text=long_text)))

    assert len(outputs) == 1
    assert captured["max_new_tokens"] == handler._estimate_max_new_tokens(long_text)
    assert captured["max_new_tokens"] > 360


def test_process_voice_clone_scales_max_tokens_for_mlx_backend(monkeypatch):
    captured = {}

    class _FakeMLXLockContext:
        def __init__(self, handler_name, timeout):
            self.handler_name = handler_name
            self.timeout = timeout

        def __enter__(self):
            return True

        def __exit__(self, exc_type, exc, tb):
            return False

    handler = object.__new__(Qwen3TTSHandler)
    handler.should_listen = Event()
    handler.cancel_scope = None
    handler.ref_audio = "TTS/ref_audio.wav"
    handler.ref_text = "Reference text."
    handler.speaker = None
    handler.instruct = None
    handler.language = "English"
    handler.xvec_only = False
    handler.parity_mode = False
    handler.non_streaming_mode = None
    handler.streaming_chunk_size = 4
    handler.max_new_tokens = 1536
    handler.blocksize = 512
    handler.backend = "mlx"
    handler.gen_kwargs = {}
    handler.queue_in = Queue()
    handler.model = SimpleNamespace(
        config=SimpleNamespace(tts_model_type="base"),
        generate=lambda **kwargs: (
            captured.update(kwargs),
            iter([(_audible_stream_chunk(), 16000, {})]),
        )[1],
    )
    handler._prepare_mlx_ref_audio = lambda ref_audio: ref_audio

    monkeypatch.setattr(qwen3_tts_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(qwen3_tts_module, "MLXLockContext", _FakeMLXLockContext)

    long_text = " ".join(["This is a deliberately long sentence for the MLX Qwen3 TTS backend."] * 12)
    outputs = list(handler.process(TTSInput(text=long_text)))

    assert len(outputs) == 1
    assert captured["max_tokens"] == handler._estimate_max_new_tokens(long_text)
    assert captured["max_tokens"] > 360
