from unittest.mock import MagicMock

import numpy as np
import pytest

from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.STT import sense_voice_handler
from speech_to_speech.STT.sense_voice_handler import SenseVoiceSTTHandler


class _FakeSenseVoiceModel:
    def generate(self, *_args, **_kwargs):
        return [{"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|> 欢迎使用 SenseVoice"}]


def _handler(device: str = "cpu"):
    handler = object.__new__(SenseVoiceSTTHandler)
    handler.model = _FakeSenseVoiceModel()
    handler.device = device
    handler.language = "auto"
    handler.gen_kwargs = {}
    handler._postprocess = lambda text: text.replace("<|zh|><|NEUTRAL|><|Speech|><|woitn|>", "")
    return handler


def test_setup_passes_explicit_generation_configuration(monkeypatch):
    model = MagicMock()
    model.generate.return_value = [{"text": "warmup"}]
    auto_model = MagicMock(return_value=model)
    postprocess = MagicMock(side_effect=lambda text: text)
    monkeypatch.setattr(sense_voice_handler, "console", MagicMock())
    monkeypatch.setattr(sense_voice_handler, "np", np)
    monkeypatch.setitem(__import__("sys").modules, "funasr", MagicMock(AutoModel=auto_model))
    utils = MagicMock()
    postprocess_module = MagicMock(rich_transcription_postprocess=postprocess)
    monkeypatch.setitem(__import__("sys").modules, "funasr.utils", utils)
    monkeypatch.setitem(__import__("sys").modules, "funasr.utils.postprocess_utils", postprocess_module)

    handler = object.__new__(SenseVoiceSTTHandler)
    handler.setup(device="cpu", language="yue", gen_kwargs={"foo": "bar"})

    auto_model.assert_called_once_with(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
    model.generate.assert_called_once()
    assert model.generate.call_args.kwargs["language"] == "yue"
    assert model.generate.call_args.kwargs["foo"] == "bar"


@pytest.mark.parametrize(("mode", "output_type"), [("progressive", PartialTranscription), ("final", Transcription)])
def test_sensevoice_emits_pipeline_transcription_events(monkeypatch, mode, output_type):
    monkeypatch.setattr(sense_voice_handler.console, "print", lambda *args, **kwargs: None)
    result = list(
        _handler().process(
            VADAudio(
                audio=np.zeros(16000, dtype=np.float32),
                mode=mode,
                turn_id="turn",
                turn_revision=3,
                created_at_s=4.5,
            )
        )
    )

    assert len(result) == 1
    assert isinstance(result[0], output_type)
    assert result[0].text == "欢迎使用 SenseVoice"
    assert result[0].turn_id == "turn"
    assert result[0].turn_revision == 3
    if mode == "final":
        assert result[0].speech_stopped_at_s == 4.5
