"""The public runtime has no optional-import or model-loading side effects."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from speech_to_speech.zerogpu import runtime


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "_worker", None)
    monkeypatch.setattr(runtime, "_settings", None)


def test_ordinary_import_does_not_import_cuda_model_or_spaces_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import speech_to_speech.zerogpu.runtime
assert not {'spaces', 'torch', 'transformers', 'nano_parakeet', 'faster_qwen3_tts'}.intersection(sys.modules)
""",
        ],
        check=True,
    )


def test_uninitialized_runtime_has_actionable_error():
    for operation in [
        runtime.prepare,
        runtime.prefetch,
        runtime.get_tokenizer,
        runtime.get_model_ids,
        runtime.get_tts_config,
    ]:
        with pytest.raises(RuntimeError, match=r"initialize\(\) before building"):
            operation()
    runtime.stop()  # Shutdown is safe before initialization.


def test_explicit_initialization_is_shared_and_does_not_silently_change_models(monkeypatch):
    worker = SimpleNamespace(
        initialize_models=Mock(),
        llm_tokenizer=object(),
        tts=SimpleNamespace(
            model_name=runtime.DEFAULT_TTS_MODEL,
            default_voice="Aiden",
            default_language="english",
            supported_languages={"en"},
            resolve_voice=lambda requested, current: requested or current,
        ),
        prepare=Mock(return_value={"models": ["stt", "llm", "tts"]}),
        stop=Mock(),
    )
    importer = Mock(return_value=worker)
    monkeypatch.setattr(runtime, "import_module", importer)
    assert runtime.initialize() is worker
    assert runtime.initialize() is worker
    importer.assert_called_once_with("speech_to_speech.zerogpu._worker")
    worker.initialize_models.assert_called_once_with(
        runtime.DEFAULT_STT_MODEL, runtime.DEFAULT_LLM_MODEL, "qwen3", None, None
    )
    assert runtime.get_tokenizer() is worker.llm_tokenizer
    assert runtime.get_tts_config()["voice"] == "Aiden"
    assert runtime.resolve_tts_voice("Vivian", "Aiden") == "Vivian"
    assert runtime.get_model_ids()["stt"] == runtime.DEFAULT_STT_MODEL
    ids = runtime.get_model_ids()
    ids["stt"] = "modified"
    assert runtime.get_model_ids()["stt"] == runtime.DEFAULT_STT_MODEL
    assert runtime.prepare()["models"] == ["stt", "llm", "tts"]
    runtime.stop()
    worker.stop.assert_called_once_with()
    with pytest.raises(RuntimeError, match="different models"):
        runtime.initialize(llm_model="other")


def test_failed_model_loading_does_not_publish_partial_runtime(monkeypatch):
    worker = SimpleNamespace(initialize_models=Mock(side_effect=RuntimeError("missing model")))
    monkeypatch.setattr(runtime, "import_module", lambda name: worker)
    with pytest.raises(RuntimeError, match="missing model"):
        runtime.initialize()
    with pytest.raises(RuntimeError, match="not initialized"):
        runtime.prepare()


def test_streaming_and_cancellation_delegate_without_buffering(monkeypatch):
    stream = iter(["first", "second"])
    worker = SimpleNamespace(
        tokens=Mock(return_value=stream),
        synthesize_chunks=Mock(return_value=stream),
    )
    monkeypatch.setattr(runtime, "_worker", worker)

    def cancelled():
        return False

    assert runtime.tokens([], {"max_new_tokens": 5}, cancelled=cancelled) is stream
    worker.tokens.assert_called_once_with([], {"max_new_tokens": 5}, cancelled=cancelled)
    assert runtime.synthesize_chunks("Hi", speaker="Aiden") is stream
    worker.synthesize_chunks.assert_called_once_with("Hi", None, "Aiden", cancelled=None)


def test_unknown_tts_backend_fails_before_importing_cuda(monkeypatch):
    importer = Mock()
    monkeypatch.setattr(runtime, "import_module", importer)
    with pytest.raises(ValueError, match="No ZeroGPU loading adapter"):
        runtime.initialize(tts_backend="unsupported")
    importer.assert_not_called()


def test_tts_backend_selection_is_independent_of_execution(monkeypatch):
    worker = SimpleNamespace(
        initialize_models=Mock(),
        tts=SimpleNamespace(
            model_name="hexgrad/Kokoro-82M",
            default_voice="bm_fable",
            default_language="en",
            supported_languages={"en", "fr"},
        ),
    )
    monkeypatch.setattr(runtime, "import_module", lambda name: worker)
    runtime.initialize(tts_backend="kokoro")
    worker.initialize_models.assert_called_once_with(
        runtime.DEFAULT_STT_MODEL, runtime.DEFAULT_LLM_MODEL, "kokoro", None, None
    )
    assert runtime.get_tts_config()["backend"] == "kokoro"
    assert runtime.get_model_ids()["tts"] == "hexgrad/Kokoro-82M"
    with pytest.raises(RuntimeError, match="different models"):
        runtime.initialize(tts_backend="qwen3")
