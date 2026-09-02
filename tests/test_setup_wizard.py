from speech_to_speech.setup.endpoints import EndpointCandidate, EndpointCapabilities
from speech_to_speech.setup.models import SetupProfile
from speech_to_speech.setup.system import GIB, SystemSnapshot
from speech_to_speech.setup.wizard import SetupWizard, run_profiled_local


class ScriptedIO:
    def __init__(self, selections, confirmations=(True, False)):
        self.selections = iter(selections)
        self.confirmations = iter(confirmations)
        self.messages = []

    def choose(self, prompt, options, default=0):
        self.messages.append(prompt)
        return next(self.selections)

    def confirm(self, prompt, default=True):
        self.messages.append(prompt)
        return next(self.confirmations)

    def print(self, message):
        self.messages.append(str(message))


def snapshot(memory=24 * GIB, free=40 * GIB):
    return SystemSnapshot("Darwin", "arm64", False, memory, "gemma", free, 1, 1, True)


def test_wizard_prefers_discovered_llm_and_small_local_speech_models(tmp_path):
    endpoint = EndpointCandidate(
        "http://127.0.0.1:8080/v1",
        ("already-running",),
        EndpointCapabilities(chat_completions=True),
    )
    io = ScriptedIO([0, 0, 0])
    saved = []
    installed = []
    wizard = SetupWizard(
        io=io,
        inspect=lambda: snapshot(),
        discover=lambda: [endpoint],
        scan_caches=lambda custom: ([], []),
        install=lambda choice: installed.append(choice.model_id),
        save=lambda profile: saved.append(profile) or tmp_path / "default.json",
        validate_endpoint=lambda *args, **kwargs: True,
    )

    assert wizard.run() == 0
    profile = saved[0]
    assert profile.pipeline["stt"] == "parakeet-tdt"
    assert profile.pipeline["tts"] == "kokoro"
    assert profile.pipeline["model_name"] == "already-running"
    assert profile.pipeline["responses_api_base_url"] == endpoint.base_url
    assert "ggml-org/gemma-4-12B-it-GGUF" not in installed


def test_wizard_defaults_to_gemma_at_24_gib_and_qwen_below_it(tmp_path):
    profiles = []
    for memory in (24 * GIB, 16 * GIB):
        io = ScriptedIO([0, 0, 0])
        wizard = SetupWizard(
            io=io,
            inspect=lambda memory=memory: snapshot(memory),
            discover=lambda: [],
            scan_caches=lambda custom: ([], []),
            install=lambda choice: None,
            save=lambda profile: profiles.append(profile) or tmp_path / "default.json",
        )
        assert wizard.run() == 0

    assert profiles[0].managed_services[0].model == "ggml-org/gemma-4-12B-it-GGUF:Q4_0"
    assert profiles[1].pipeline["model_name"] == "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_wizard_prompts_for_auth_only_after_protected_endpoint_is_selected(tmp_path):
    from speech_to_speech.setup.models import CredentialRef

    class FakeKeychain:
        def __init__(self):
            self.prompted = []

        def prompt_and_store(self, url):
            self.prompted.append(url)
            return CredentialRef("speech-to-speech", "endpoint-test")

        def get(self, reference):
            return "runtime-key"

    keychain = FakeKeychain()
    validations = []
    protected = EndpointCandidate("http://127.0.0.1:9000/v1", requires_auth=True)
    profiles = []
    wizard = SetupWizard(
        io=ScriptedIO([0, 0, 0]),
        inspect=snapshot,
        discover=lambda: [protected],
        scan_caches=lambda custom: ([], []),
        install=lambda choice: None,
        save=lambda profile: profiles.append(profile) or tmp_path / "default.json",
        keychain=keychain,
        validate_endpoint=lambda *args, **kwargs: validations.append((args, kwargs)) or True,
    )

    assert wizard.run() == 0
    assert keychain.prompted == [protected.base_url]
    assert validations[0][1]["api_key"] == "runtime-key"
    assert profiles[0].credentials["llm"].account == "endpoint-test"


def test_profile_launch_resolves_secret_only_in_memory():
    profile = SetupProfile(
        pipeline={
            "stt": "parakeet-tdt",
            "llm_backend": "chat-completions",
            "tts": "kokoro",
            "responses_api_base_url": "http://127.0.0.1:8080/v1",
        }
    )
    from speech_to_speech.setup.models import CredentialRef

    profile.credentials["llm"] = CredentialRef("speech-to-speech", "endpoint-test")
    calls = []

    run_profiled_local(
        profile,
        credential_getter=lambda ref: "runtime-secret",
        pipeline_runner=lambda command, args: calls.append((command, args)),
    )

    assert calls[0][0] == "local"
    assert "runtime-secret" in calls[0][1]
    assert "runtime-secret" not in repr(profile)
