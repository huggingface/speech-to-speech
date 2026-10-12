import pytest

from speech_to_speech.LLM.language_model import LanguageModelHandler, VisionLanguageModelHandler


@pytest.mark.parametrize("handler_type", [LanguageModelHandler, VisionLanguageModelHandler])
@pytest.mark.parametrize("available_device", ["cuda", "cpu"])
def test_local_transformers_resolves_auto_device_before_loading(handler_type, available_device, monkeypatch):
    monkeypatch.setattr("speech_to_speech.utils.utils._is_device_available", lambda device: device == available_device)
    loaded_devices = []
    monkeypatch.setattr(
        handler_type, "_load_model", lambda self, model, device, dtype, kwargs: loaded_devices.append(device)
    )
    monkeypatch.setattr(LanguageModelHandler, "warmup", lambda self: None)
    handler = object.__new__(handler_type)

    handler.setup(device="auto", backend="transformers")

    assert loaded_devices == [available_device]
    assert handler.device == available_device


def test_local_mlx_does_not_probe_torch_devices(monkeypatch):
    monkeypatch.setattr("speech_to_speech.utils.utils._is_device_available", lambda device: pytest.fail("Torch probe"))
    loaded_devices = []
    monkeypatch.setattr(
        LanguageModelHandler, "_load_model", lambda self, model, device, dtype, kwargs: loaded_devices.append(device)
    )
    monkeypatch.setattr(LanguageModelHandler, "warmup", lambda self: None)
    handler = object.__new__(LanguageModelHandler)

    handler.setup(device="mps", backend="mlx")

    assert loaded_devices == ["mps"]
