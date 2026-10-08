import copy
import math
import subprocess
import sys
from dataclasses import fields

import pytest

from speech_to_speech.config import ConfigurationError, load_config, resolve_config, validate_config

BASE = """schema_version: 1
blocks:
  vad: {kind: vad, backend: silero}
  stt: {kind: stt, backend: openai}
  llm: {kind: llm, backend: chat-completions}
  tts: {kind: tts, backend: openai}
pipelines:
  primary:
    num_pipelines: 2
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
  secondary:
    stages: {vad: vad, stt: stt, llm: llm, tts: tts}
"""


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    return load_config(path)


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return load_config(path)


def test_load_resolve_selection_defaults_and_independence(document):
    validate_config(document)
    original = copy.deepcopy(document.data)
    result = resolve_config(document, names=["secondary", "primary"], environ={})
    assert list(result.pipelines) == ["secondary", "primary"]
    assert result.pipelines["primary"].num_pipelines == 2
    assert result.pipelines["secondary"].num_pipelines == 1
    assert math.isinf(result.pipelines["primary"].stages["vad"].settings["max_speech_ms"])
    assert result.runtime["server"]["port"] == 8765
    first = result.pipelines["primary"].stages["llm"]
    first.settings["model_name"] = "changed"
    first.selection.config["gen_kwargs"]["custom"] = 42
    assert result.pipelines["secondary"].stages["llm"].settings["model_name"] != "changed"
    assert "custom" not in resolve_config(document).pipelines["primary"].stages["llm"].selection.config["gen_kwargs"]
    assert document.data == original


@pytest.mark.parametrize("names", [[], "primary", ["primary", "primary"], ["unknown"]])
def test_invalid_selection(document, names):
    with pytest.raises(ConfigurationError):
        resolve_config(document, names=names)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=2),
        lambda d: d.update(variables={}),
        lambda d: d["blocks"].update({"bad.id": {"kind": "vad", "backend": "silero"}}),
        lambda d: d["blocks"]["vad"].update(kind="other"),
        lambda d: d["blocks"]["stt"].update(backend="missing"),
        lambda d: d["blocks"]["stt"].update(settings={"api_key": "secret"}),
        lambda d: d["blocks"]["vad"].update(settings={"sample_rate": True}),
        lambda d: d["blocks"]["vad"].update(settings={"sample_rate": "16000"}),
        lambda d: d["blocks"]["vad"].update(settings={"smart_turn": None}),
        lambda d: d["pipelines"]["primary"]["stages"].pop("stt"),
        lambda d: d["pipelines"]["primary"]["stages"].update(stt="vad"),
        lambda d: d["pipelines"]["primary"].update(num_pipelines=0),
        lambda d: d["pipelines"]["primary"].update(num_pipelines=True),
        lambda d: d["pipelines"]["primary"].update(options={"log_level": "info"}),
        lambda d: d.update(runtime={"client": {"tool_executor": "foo"}}),
    ],
)
def test_invalid_schema(document, change):
    change(document.data)
    with pytest.raises(ConfigurationError):
        validate_config(document)


@pytest.mark.parametrize(
    "text",
    [
        BASE + "schema_version: 1\n",
        BASE.replace("backend: silero", "backend: silero, backend: firered"),
        BASE.replace("num_pipelines: 2", "num_pipelines: 2\n    num_pipelines: 3"),
        BASE.replace(
            "backend: openai}", "backend: openai, settings: {openai_stt_api_key: x, openai_stt_api_key: y}}", 1
        ),
        BASE.replace(
            "backend: chat-completions}",
            "backend: chat-completions, settings: {responses_api_gen_kwargs: {x: 1, x: 2}}}",
        ),
        BASE + "---\n{}",
        "",
        "[1, 2]",
        "schema_version: !!int 1",
        BASE.replace("backend: silero", "backend: &anchor silero"),
        BASE.replace("backend: silero", "backend: *anchor"),
        BASE.replace("backend: silero", "backend: silero, <<: {}"),
        BASE + "7: value\n",
    ],
)
def test_parse_rejections(tmp_path, text):
    with pytest.raises(ConfigurationError):
        write(tmp_path, text)


def test_environment_selected_only_and_safe_representations(document):
    document.data["blocks"]["stt"]["settings"] = {"openai_stt_api_key": {"env": "KEY"}}
    document.data["blocks"]["unused"] = {
        "kind": "stt",
        "backend": "openai",
        "settings": {"openai_stt_api_key": {"env": "INACTIVE"}},
    }
    document.data["runtime"] = {"client": {"api_key": {"env": "CLIENT"}}}
    validate_config(document)
    result = resolve_config(document, environ={"KEY": "SECRET_MARKER"})
    assert result.pipelines["primary"].stages["stt"].settings["openai_stt_api_key"] == "SECRET_MARKER"
    assert result.client is None
    for obj in [document, result, result.pipelines["primary"], result.pipelines["primary"].stages["stt"]]:
        assert "SECRET_MARKER" not in repr(obj)
        assert "INACTIVE" not in repr(obj)
    with pytest.raises(ConfigurationError) as error:
        resolve_config(document, environ={})
    assert "KEY" not in str(error.value)
    with pytest.raises(ConfigurationError):
        resolve_config(document, environ={"KEY": ""}, include_client=True)


@pytest.mark.parametrize(
    ("value", "accepted"),
    [("true", True), ("false", True), ("True", False), ("1", False), (" true", False), ("", False)],
)
def test_environment_boolean(document, value, accepted):
    document.data["blocks"]["vad"]["settings"] = {"smart_turn": {"env": "VALUE"}}
    if accepted:
        result = resolve_config(document, environ={"VALUE": value})
        assert result.pipelines["primary"].stages["vad"].settings["smart_turn"] is (value == "true")
    else:
        with pytest.raises(ConfigurationError):
            resolve_config(document, environ={"VALUE": value})


@pytest.mark.parametrize(
    ("value", "accepted"),
    [("+2", True), ("2", True), ("-1", False), ("0", False), (" 2", False), ("0x2", False), ("2.0", False)],
)
def test_environment_count(document, value, accepted):
    document.data["pipelines"]["primary"]["num_pipelines"] = {"env": "COUNT"}
    if accepted:
        assert resolve_config(document, environ={"COUNT": value}).pipelines["primary"].num_pipelines == 2
    else:
        with pytest.raises(ConfigurationError):
            resolve_config(document, environ={"COUNT": value})


def test_literal_values_and_paths(document, monkeypatch, tmp_path):
    document.data["blocks"]["vad"]["settings"] = {
        "max_speech_ms": None,
        "smart_turn": False,
        "short_segment_merge_ms": 0,
        "smart_turn_model_path": "assets/model.onnx",
    }
    document.data["blocks"]["stt"]["settings"] = {
        "openai_stt_api_key": "",
        "openai_stt_base_url": "https://example.com/v1",
    }
    monkeypatch.chdir(tmp_path.parent)
    stage = resolve_config(document).pipelines["primary"].stages
    assert stage["vad"].settings["smart_turn"] is False
    assert stage["vad"].settings["short_segment_merge_ms"] == 0
    assert stage["vad"].settings["smart_turn_model_path"] == str(document.path.parent / "assets/model.onnx")
    assert stage["stt"].settings["openai_stt_api_key"] == ""
    assert stage["stt"].settings["openai_stt_base_url"] == "https://example.com/v1"


def test_bypass_and_capabilities(document):
    document.data["blocks"]["stt"]["backend"] = "none"
    validate_config(document)
    document.data["pipelines"]["primary"]["options"] = {"diarization_model_name": "model"}
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data["pipelines"]["primary"].pop("options")
    document.data["blocks"]["llm"]["backend"] = "responses-api"
    with pytest.raises(ConfigurationError):
        validate_config(document)


def test_yaml_scalars_and_limits(tmp_path):
    for value in ["on", "off", "yes", "2026-10-08", "0123"]:
        doc = write(
            tmp_path,
            BASE.replace("backend: openai}", "backend: openai, settings: {openai_stt_api_key: " + value + "}}", 1),
        )
        assert resolve_config(doc).pipelines["primary"].stages["stt"].settings["openai_stt_api_key"] == value
    for value in [".nan", ".inf", "1e999"]:
        with pytest.raises(ConfigurationError):
            write(tmp_path, BASE.replace("num_pipelines: 2", f"num_pipelines: {value}"))
    with pytest.raises(ConfigurationError):
        write(tmp_path, "x: " + "[" * 32 + "0" + "]" * 32)
    with pytest.raises(ConfigurationError):
        write(tmp_path, "#" * (1024 * 1024 + 1))


def test_io_and_secret_errors(tmp_path, document):
    for suffix in [".json", ".txt"]:
        path = tmp_path / ("config" + suffix)
        path.write_text(BASE)
        with pytest.raises(ConfigurationError):
            load_config(path)
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(ConfigurationError):
        load_config(path)
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigurationError) as error:
        write(tmp_path, "password: SECRET_MARKER\ninvalid: [")
    assert "SECRET_MARKER" not in str(error.value)
    document.data["blocks"]["vad"]["settings"] = {"sample_rate": "SECRET_MARKER"}
    with pytest.raises(ConfigurationError) as error:
        validate_config(document)
    assert "SECRET_MARKER" not in str(error.value)


def test_registry_equivalence_and_scope_coverage(document):
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.backend_registry import LLM_BACKENDS, STT_BACKENDS, TTS_BACKENDS
    from speech_to_speech.config import _MODULE_PROCESS, _MODULE_SELECTORS

    assert set(_MODULE_PROCESS) | set(_MODULE_SELECTORS) | set(
        resolve_config(document).pipelines["primary"].options
    ) | {"num_pipelines"} == {f.name for f in fields(ModuleArguments)}
    for kind, registry in [("stt", STT_BACKENDS), ("llm", LLM_BACKENDS), ("tts", TTS_BACKENDS)]:
        for name, spec in registry.items():
            doc = copy.deepcopy(document)
            doc.data["blocks"][kind]["backend"] = name
            if kind == "stt" and name == "none":
                doc.data["blocks"]["llm"]["backend"] = "chat-completions"
            result = resolve_config(doc)
            assert result.pipelines["primary"].stages[kind].selection.config == spec.normalize(spec.config_type())


def test_import_boundaries_and_missing_parser(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == "yaml" or fullname.startswith("yaml."):
            raise ImportError("blocked")
        if fullname.endswith("s2s_pipeline") or (fullname.startswith("speech_to_speech.") and "_handler" in fullname):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from speech_to_speech.config import load_config
assert "speech_to_speech.backend_registry" not in sys.modules
try:
    load_config(sys.argv[1])
except ImportError as exc:
    assert "speech-to-speech[config]" in str(exc)
else:
    raise AssertionError("missing dependency accepted")
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_error_context_and_long_numbers(tmp_path, document):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"SECRET_MARKER\xff")
    with pytest.raises(ConfigurationError) as error:
        load_config(path)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    with pytest.raises(ConfigurationError):
        write(tmp_path, BASE.replace("num_pipelines: 2", "num_pipelines: " + "1" * 5000))
    document.data["pipelines"]["primary"]["num_pipelines"] = {"env": "COUNT"}
    with pytest.raises(ConfigurationError):
        resolve_config(document, environ={"COUNT": "1" * 5000})


def test_inactive_client_literal_validation_and_source(document):
    document.data["runtime"] = {"client": {"playback_buffer_ms": -1}}
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data["runtime"]["client"] = {"tool_module": {"env": "TOOLS"}}
    result = resolve_config(document, include_client=True, environ={"TOOLS": "my_tools"})
    assert result.client["tool_module"] == "my_tools"
    assert result.sources[("runtime", "client", "tool_module")] == "environment"


@pytest.mark.parametrize("value", ["1e-2", "+.2", "-0.0"])
def test_environment_float(document, value):
    document.data["blocks"]["vad"]["settings"] = {"thresh": {"env": "VALUE"}}
    assert resolve_config(document, environ={"VALUE": value}).pipelines["primary"].stages["vad"].settings[
        "thresh"
    ] == float(value)


@pytest.mark.parametrize("value", ["NaN", "inf", "1e999", " 1.0", "1.0 ", "0x1", ""])
def test_invalid_environment_float(document, value):
    document.data["blocks"]["vad"]["settings"] = {"thresh": {"env": "VALUE"}}
    with pytest.raises(ConfigurationError):
        resolve_config(document, environ={"VALUE": value})


def test_literals_choices_and_inactive_invalid_references(document):
    document.data["blocks"]["unused"] = {
        "kind": "tts",
        "backend": "qwen3",
        "settings": {"qwen3_tts_backend": "unknown"},
    }
    with pytest.raises(ConfigurationError):
        resolve_config(document, names=["primary"])
    document.data["blocks"]["unused"]["settings"] = {"qwen3_tts_backend": {"env": "bad.name"}}
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data["blocks"]["unused"]["settings"] = {"qwen3_tts_backend": {"env": "INACTIVE"}}
    validate_config(document)
    resolve_config(document, names=["primary"], environ={})


def test_open_provider_dictionary_is_not_interpolated(document):
    document.data["blocks"]["stt"]["backend"] = "mlx-audio-whisper"
    document.data["blocks"]["stt"]["settings"] = {
        "mlx_audio_whisper_gen_kwargs": {"env": "NOT_READ", "custom": [0, False, None]}
    }
    result = resolve_config(document, environ={})
    stage = result.pipelines["primary"].stages["stt"]
    assert result.sources[("blocks", "stt", "settings", "mlx_audio_whisper_gen_kwargs")] == "file"
    assert stage.selection.config["gen_kwargs"] == {"env": "NOT_READ", "custom": [0, False, None]}


def test_runtime_guard_fresh_process(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(BASE)
    script = """
import importlib.abc, sys, threading, socket
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.endswith("s2s_pipeline") or (fullname.startswith("speech_to_speech.") and "_handler" in fullname):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
def blocked(*args, **kwargs):
    raise AssertionError("runtime resource created")
threading.Thread.start = blocked
socket.socket.bind = blocked
import nltk, openai
nltk.download = blocked
openai.OpenAI.__init__ = blocked
from speech_to_speech.config import load_config, validate_config, resolve_config
assert "speech_to_speech.backend_registry" not in sys.modules
document = load_config(sys.argv[1])
assert "speech_to_speech.backend_registry" not in sys.modules
validate_config(document)
resolve_config(document, environ={})
assert "speech_to_speech.s2s_pipeline" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("threshold", [0, 1])
def test_diarization_flag_composition_and_threshold(document, threshold):
    document.data["pipelines"]["primary"]["options"] = {"diarization": True, "diarization_threshold": threshold}
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data["pipelines"]["primary"]["options"] = {"diarization": True}
    document.data["blocks"]["stt"]["backend"] = "none"
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data["pipelines"]["primary"]["options"] = {"diarization": {"env": "DIARIZATION"}}
    validate_config(document)
    with pytest.raises(ConfigurationError):
        resolve_config(document, environ={"DIARIZATION": "true"})
    resolve_config(document, environ={"DIARIZATION": "false"})


def test_duplicate_diagnostic_and_huge_numeric_type(tmp_path, document):
    with pytest.raises(ConfigurationError, match="Duplicate") as error:
        write(tmp_path, BASE + "schema_version: 1\n")
    assert error.value.__context__ is None
    document.data["blocks"]["vad"]["settings"] = {"thresh": 10**1000}
    with pytest.raises(ConfigurationError):
        validate_config(document)


def test_schema_references_paths_and_provider_error_redaction(document, monkeypatch):
    document.data["blocks"]["vad"]["settings"] = {"smart_turn_model_path": "~/model.onnx"}
    monkeypatch.setenv("HOME", "/tmp/config-home")
    assert (
        resolve_config(document).pipelines["primary"].stages["vad"].settings["smart_turn_model_path"]
        == "/tmp/config-home/model.onnx"
    )
    document.data["blocks"]["stt"]["backend"] = "mlx-audio-whisper"
    document.data["blocks"]["stt"]["settings"] = {"mlx_audio_whisper_gen_kwargs": {"SECRET_MARKER": object()}}
    with pytest.raises(ConfigurationError) as error:
        validate_config(document)
    assert "SECRET_MARKER" not in str(error.value)


def test_collection_environment_and_failed_home_expansion(document):
    document.data["runtime"] = {"client": {"tools": {"env": "TOOLS"}}}
    with pytest.raises(ConfigurationError):
        validate_config(document)
    document.data.pop("runtime")
    document.data["blocks"]["vad"]["settings"] = {"smart_turn_model_path": "~SECRET_MARKER/model.onnx"}
    with pytest.raises(ConfigurationError) as error:
        resolve_config(document)
    assert "SECRET_MARKER" not in str(error.value)
    assert error.value.__context__ is None


def test_config_path_home_expansion_is_safe():
    with pytest.raises(ConfigurationError) as error:
        load_config("~SECRET_MARKER/config.yaml")
    assert "SECRET_MARKER" not in str(error.value)
    assert error.value.__context__ is None


def test_parser_size_and_depth_boundaries(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(BASE + "#" * (1024 * 1024 - len(BASE.encode())))
    validate_config(load_config(path))
    document = write(tmp_path, "x: " + "[" * 31 + "0" + "]" * 31)
    assert "x" in document.data


def test_source_metadata_explicit_and_default_values(document):
    document.data["blocks"]["vad"]["settings"] = {"smart_turn": False}
    sources = resolve_config(document).sources
    assert sources[("blocks", "vad", "settings", "smart_turn")] == "file"
    assert sources[("blocks", "vad", "settings", "thresh")] == "default"
    assert sources[("pipelines", "primary", "num_pipelines")] == "file"
    assert sources[("pipelines", "secondary", "num_pipelines")] == "default"


def test_explicit_scope_inventory():
    from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.realtime_server_arguments import RealtimeServerArguments
    from speech_to_speech.config import _MODULE_PIPELINE, _MODULE_PROCESS, _MODULE_SELECTORS

    assert _MODULE_PROCESS | _MODULE_PIPELINE | _MODULE_SELECTORS | {"num_pipelines"} == {
        f.name for f in fields(ModuleArguments)
    }
    assert {"enable_visemes", "stv_model_name", "stv_device"} <= _MODULE_PIPELINE
    assert not _MODULE_PROCESS & _MODULE_PIPELINE
    assert {f.name for f in fields(RealtimeServerArguments)} == {"host", "port"}
    assert {f.name for f in fields(RealtimeAudioClientConfig)} == {
        "url",
        "model",
        "api_key",
        "send_rate",
        "recv_rate",
        "playback_buffer_ms",
        "chunk_size",
        "input_device",
        "output_device",
        "instructions",
        "voice",
        "print_json",
        "block_mic_during_playback",
        "log_transcripts",
        "connection_retry_timeout_s",
        "tools",
        "tool_executor",
        "tool_response_create",
    }


def test_inherited_viseme_options_defaults_sources_and_inactive_environment(document):
    from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
    from speech_to_speech.arguments_classes.w2v_stv_arguments import Wav2Vec2STVHandlerArguments

    defaults = Wav2Vec2STVHandlerArguments()
    inherited = {field.name: field for field in fields(ModuleArguments)}
    result = resolve_config(document, names=["primary"], environ={})
    for field in fields(defaults):
        assert result.pipelines["primary"].options[field.name] == getattr(defaults, field.name)
        assert result.sources[("pipelines", "primary", "options", field.name)] == "default"
        assert inherited[field.name].metadata == field.metadata
    document.data["pipelines"]["primary"]["options"] = {
        "enable_visemes": {"env": "ENABLED"},
        "stv_model_name": {"env": "MODEL"},
        "stv_device": {"env": "DEVICE"},
    }
    document.data["pipelines"]["secondary"]["options"] = {
        "enable_visemes": {"env": "UNUSED_ENABLED"},
        "stv_model_name": {"env": "UNUSED_MODEL"},
        "stv_device": {"env": "UNUSED_DEVICE"},
    }
    validate_config(document)
    result = resolve_config(document, names=["primary"], environ={"ENABLED": "false", "MODEL": "", "DEVICE": "cpu"})
    assert result.pipelines["primary"].options["enable_visemes"] is False
    assert result.pipelines["primary"].options["stv_model_name"] == ""
    assert result.pipelines["primary"].options["stv_device"] == "cpu"
    for field in fields(defaults):
        assert result.sources[("pipelines", "primary", "options", field.name)] == "environment"
    with pytest.raises(ConfigurationError):
        resolve_config(document, names=["secondary"], environ={})
