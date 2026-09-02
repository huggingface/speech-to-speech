import pytest

from speech_to_speech.setup.models import ManagedService
from speech_to_speech.setup.services import ManagedServiceRunner


class FakeProcess:
    def __init__(self, poll_results=None):
        self.poll_results = iter(poll_results or [None])
        self.terminated = False
        self.waited = False

    def poll(self):
        return next(self.poll_results, None)

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        self.waited = True


def test_managed_llama_uses_dynamic_loopback_port_and_expected_model(monkeypatch):
    process = FakeProcess()
    calls = []
    runner = ManagedServiceRunner(
        llama_server="/runtime/llama-server",
        popen=lambda command, **kwargs: calls.append((command, kwargs)) or process,
        port_picker=lambda: 54321,
        readiness=lambda url: True,
    )
    spec = ManagedService("ggml-org/gemma-4-12B-it-GGUF:Q4_0")

    managed = runner.start(spec)

    command = calls[0][0]
    assert command[:5] == ["/runtime/llama-server", "--host", "127.0.0.1", "--port", "54321"]
    assert command[command.index("-hf") + 1] == spec.model
    assert managed.base_url == "http://127.0.0.1:54321/v1"
    managed.stop()
    assert process.terminated is True
    assert process.waited is True


def test_managed_service_reports_early_crash():
    process = FakeProcess([7])
    runner = ManagedServiceRunner(
        llama_server="llama-server",
        popen=lambda *args, **kwargs: process,
        port_picker=lambda: 54321,
        readiness=lambda url: False,
        sleep=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="exited with status 7"):
        runner.start(ManagedService("org/model:Q4"))


def test_managed_llama_uses_installed_model_path_when_available():
    process = FakeProcess()
    commands = []
    runner = ManagedServiceRunner(
        llama_server="llama-server",
        popen=lambda command, **kwargs: commands.append(command) or process,
        port_picker=lambda: 54321,
        readiness=lambda url: True,
    )

    runner.start(ManagedService("org/model:Q4", model_path="/models/model.Q4_0.gguf"))

    assert "-hf" not in commands[0]
    assert commands[0][commands[0].index("-m") + 1] == "/models/model.Q4_0.gguf"
