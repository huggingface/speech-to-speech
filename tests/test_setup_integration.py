from speech_to_speech.s2s_pipeline import parse_arguments
from speech_to_speech.setup.catalog import KOKORO, PARAKEET
from speech_to_speech.setup.endpoints import EndpointCandidate
from speech_to_speech.setup.models import CredentialRef
from speech_to_speech.setup.profiles import load_profile, save_profile
from speech_to_speech.setup.system import GIB, CachedModel, SystemSnapshot
from speech_to_speech.setup.wizard import SetupWizard, run_profiled_local


class IntegrationIO:
    def __init__(self):
        self.confirmations = iter((True, False))

    def choose(self, prompt, options, default=0):
        return default

    def confirm(self, prompt, default=True):
        return next(self.confirmations)

    def print(self, message):
        pass


class FakeKeychain:
    reference = CredentialRef("endpoint-integration")

    def prompt(self, url):
        return "integration-secret"

    def store(self, url, secret):
        assert secret == "integration-secret"
        return self.reference

    def get(self, reference):
        assert reference == self.reference
        return "integration-secret"


def test_discover_choose_save_resolve_and_launch_without_persisting_secret(tmp_path):
    profile_path = tmp_path / "profiles" / "default.json"
    installed = []
    validations = []
    endpoint = EndpointCandidate("http://127.0.0.1:61234/v1", requires_auth=True)
    wizard = SetupWizard(
        io=IntegrationIO(),
        inspect=lambda: SystemSnapshot(24 * GIB, 30 * GIB, 1, 1, True),
        discover=lambda: [endpoint],
        scan_caches=lambda custom: ([CachedModel(PARAKEET.model_id, PARAKEET.estimated_bytes, tmp_path)], []),
        install=lambda choice: installed.append(choice.model_id),
        save=lambda profile: save_profile(profile, profile_path),
        keychain=FakeKeychain(),
        validate_endpoint=lambda *args, **kwargs: validations.append((args, kwargs)) or True,
    )

    assert wizard.run() == 0
    assert installed == [KOKORO.model_id]
    assert validations[0][0][0] == endpoint.base_url
    assert "integration-secret" not in profile_path.read_text()
    assert load_profile(profile_path).pipeline["parakeet_tdt_model_name"] == str(tmp_path)

    launched = []
    run_profiled_local(
        load_profile(profile_path),
        credential_getter=FakeKeychain().get,
        pipeline_runner=lambda command, args: launched.append((command, args)),
    )
    assert launched[0][0] == "local"
    assert "integration-secret" in launched[0][1]
    parsed = parse_arguments(launched[0][1], command="local")
    assert parsed.stt_backend.config["model_name"] == str(tmp_path)
    assert parsed.llm_backend.config["base_url"] == endpoint.base_url
    assert parsed.llm_backend.config["api_key"] == "integration-secret"
