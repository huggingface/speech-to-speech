import sys
from threading import Event
from types import SimpleNamespace

import pytest

from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler


@pytest.mark.parametrize(
    ("setup_kwargs", "expected_language"),
    [
        ({}, "english"),
        ({"language": "french_24l"}, "french_24l"),
    ],
)
def test_pocket_tts_setup_loads_language(monkeypatch, setup_kwargs, expected_language):
    loaded_languages = []

    fake_model = SimpleNamespace(
        to=lambda *args, **kwargs: None,
        get_state_for_audio_prompt=lambda *args, **kwargs: object(),
        sample_rate=24000,
    )

    def fake_load_model(*, language):
        loaded_languages.append(language)
        return fake_model

    fake_pocket_tts = SimpleNamespace(
        TTSModel=SimpleNamespace(load_model=fake_load_model),
    )

    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    handler = object.__new__(PocketTTSHandler)

    handler.setup(
        should_listen=Event(),
        **setup_kwargs,
    )

    assert loaded_languages == [expected_language]


def test_pocket_tts_custom_config(monkeypatch):
    calls = []
    model = SimpleNamespace(sample_rate=24000, get_state_for_audio_prompt=lambda voice: voice)
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts",
        SimpleNamespace(
            TTSModel=SimpleNamespace(
                load_model=lambda **kwargs: calls.append(kwargs) or model,
            )
        ),
    )
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    handler.setup(Event(), model_name="/models/custom.yaml")
    assert calls == [{"config": "/models/custom.yaml"}]


def test_farsi_requires_audio_reference():
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    with pytest.raises(ValueError, match="reference audio"):
        handler.setup(Event(), model_name="mehdi-hf/pocket-tts-farsi-v2")


def test_farsi_setup_loads_config_and_trims_reference(monkeypatch):
    import huggingface_hub
    import torch

    from speech_to_speech.TTS import pocket_tts_handler

    calls = []
    audio = torch.ones(2, 6 * 16000)
    model = SimpleNamespace(
        sample_rate=24000,
        get_state_for_audio_prompt=lambda ref: calls.append(ref) or "state",
        _generate_audio_stream_short_text=lambda **kwargs: iter(()),
        capitalize_first_letter=False,
        append_terminal_punctuation=False,
        pad_with_spaces_for_short_inputs=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts",
        SimpleNamespace(
            TTSModel=SimpleNamespace(
                load_model=lambda **kwargs: calls.append(kwargs) or model,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pocket_tts.utils.config",
        SimpleNamespace(
            Config=SimpleNamespace(
                model_fields={
                    "capitalize_first_letter": None,
                    "append_terminal_punctuation": None,
                    "pad_with_spaces_for_short_inputs": None,
                },
            )
        ),
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **kwargs: calls.append(kwargs) or "/cache/model.yaml"
    )
    monkeypatch.setitem(sys.modules, "pocket_tts.data.audio", SimpleNamespace(audio_read=lambda path: (audio, 16000)))
    monkeypatch.setitem(sys.modules, "pocket_tts.utils.utils", SimpleNamespace(download_if_necessary=lambda path: path))

    def convert(ref, source_rate, target_rate, channels):
        assert ref.shape == (2, 5 * 16000)
        assert (source_rate, target_rate, channels) == (16000, 24000, 1)
        return torch.ones(1, 5 * 24000)

    monkeypatch.setitem(sys.modules, "pocket_tts.data.audio_utils", SimpleNamespace(convert_audio=convert))
    monkeypatch.setattr(pocket_tts_handler, "PersianPhonemizer", lambda device: "phonemizer")
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    handler.setup(Event(), voice="reference.wav", model_name="mehdi-hf/pocket-tts-farsi-v2")
    assert calls[0] == {"repo_id": "mehdi-hf/pocket-tts-farsi-v2", "filename": "model.yaml"}
    assert calls[1] == {"config": "/cache/model.yaml", "eos_threshold": -2.0}
    assert calls[2].shape == (1, 120000)
    assert handler.voice_state == "state"
    assert handler.max_tokens == 18
    assert handler.phonemizer == "phonemizer"


def test_farsi_rejects_incompatible_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=object))
    monkeypatch.setitem(
        sys.modules, "pocket_tts.utils.config", SimpleNamespace(Config=SimpleNamespace(model_fields={}))
    )
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    with pytest.raises(RuntimeError, match="fork"):
        handler.setup(Event(), voice="reference.wav", model_name="mehdi-hf/pocket-tts-farsi-v2")


def _farsi_handler():
    import torch

    calls = []
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    handler.cancel_scope = None
    handler.speculative_turns = None
    handler.voice_state = "voice"
    handler.max_tokens = 3
    handler.sample_rate = 16000
    handler.blocksize = 512
    handler.model = SimpleNamespace(
        sample_rate=24000,
        flow_lm=SimpleNamespace(
            conditioner=SimpleNamespace(
                tokenizer=SimpleNamespace(sp=SimpleNamespace(encode=lambda text: text.split(), unk_id=lambda: -1))
            )
        ),
        _generate_audio_stream_short_text=lambda *, model_state, text_to_generate, **kwargs: (
            calls.append((model_state, text_to_generate, kwargs)) or iter([torch.full((2400,), 0.1)])
        ),
        generate_audio_stream=lambda *args, **kwargs: pytest.fail(
            "Farsi phonemes must bypass the public text splitter"
        ),
    )
    return handler, calls


def test_farsi_sentence_conversion_and_audio_output():
    import numpy as np

    from speech_to_speech.pipeline.messages import TTSInput

    handler, calls = _farsi_handler()
    converted = []
    handler.phonemizer = lambda text: converted.append(text) or "salAm hAle1 SomA"
    output = list(handler.process(TTSInput(text="سلام. حال شما؟")))
    assert converted == ["سلام", "حال شما"]
    assert [call[1] for call in calls] == ["salAm hAle SomA", "salAm hAle SomA"]
    assert all(call[2]["copy_state"] for call in calls)
    assert all(block.dtype == np.int16 and block.shape == (512,) for block in output)
    assert any(np.any(block) for block in output)
    assert sum(len(block) for block in output) >= 4000  # Sentence pause is included.


def test_phoneme_chunking_preserves_ezafe():
    handler, _ = _farsi_handler()
    assert list(handler._phoneme_chunks("a b c d e f")) == ["a b c", "d e f"]
    assert list(handler._phoneme_chunks("a b c1 d e f")) == ["a b", "c d e", "f"]


def test_cancellation_during_phonemization_skips_synthesis():
    handler, calls = _farsi_handler()
    state = {"cancelled": False}
    handler.cancel_scope = SimpleNamespace(is_stale=lambda generation: state["cancelled"])

    def phonemize(text):
        state["cancelled"] = True
        return "salAm"

    handler.phonemizer = phonemize
    assert list(handler._generate_audio("سلام", 0)) == []
    assert calls == []


def test_empty_phonemes_skip_synthesis():
    handler, calls = _farsi_handler()
    handler.phonemizer = lambda text: ""
    assert list(handler._generate_audio("!", None)) == []
    assert calls == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("تا سال ۲۰۳۰", "تا سال دو هزار و سی"),
        ("تا سال ٢٠٣٠", "تا سال دو هزار و سی"),
        ("تا سال 2030", "تا سال دو هزار و سی"),
        ("كِتاب يكي", "کتاب یکی"),
        ("۳٫۵٪", "سه ممیز پنج دهم درصد"),
        ("۰۹۱۲", "صفر نه یک دو"),
        ("می‌رود", "می‌رود"),
    ],
)
def test_persian_normalization(text, expected):
    from speech_to_speech.TTS.persian_normalization import normalize

    assert normalize(text) == expected


def test_phonemizer_normalizes_before_g2p(monkeypatch):
    import torch

    from speech_to_speech.TTS.pocket_tts_farsi import PersianPhonemizer

    inputs = []

    class Encoded(dict):
        def to(self, device):
            return self

    phonemizer = PersianPhonemizer.__new__(PersianPhonemizer)
    phonemizer.device = "cpu"
    phonemizer.tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: ["s/lam hale1 $oma cetor @ast"])

    class Tokenizer:
        def __call__(self, text, **kwargs):
            inputs.append((text, kwargs))
            return Encoded(input_ids=torch.zeros((1, 10)))

        batch_decode = phonemizer.tokenizer.batch_decode

    phonemizer.tokenizer = Tokenizer()
    phonemizer.model = SimpleNamespace(generate=lambda **kwargs: torch.zeros((1, 20)))
    assert phonemizer("سلام ۲۰۳۰؟") == "salAm hAle1 SomA Cetor ?Ast"
    assert inputs[0][0] == ["سلام دو هزار و سی"]
    assert inputs[0][1]["add_special_tokens"] is False
    assert phonemizer("!") == ""
    assert len(inputs) == 1


def test_cancellation_discards_buffered_sentence_audio():
    import torch

    from speech_to_speech.pipeline.messages import TTSInput

    handler, _ = _farsi_handler()
    state = {"cancelled": False, "sentences": 0}
    handler.cancel_scope = SimpleNamespace(generation=0, is_stale=lambda generation: state["cancelled"])
    handler.blocksize = 4096
    handler.model._generate_audio_stream_short_text = lambda **kwargs: iter([torch.ones(2400) * 0.1])

    def phonemize(text):
        state["sentences"] += 1
        if state["sentences"] == 2:
            state["cancelled"] = True
        return "salAm"

    handler.phonemizer = phonemize
    assert list(handler.process(TTSInput(text="سلام. سلام."))) == []


@pytest.mark.parametrize("input_length, output_length", [(513, 10), (10, 512)])
def test_phonemizer_rejects_length_limits(input_length, output_length):
    import torch

    from speech_to_speech.TTS.pocket_tts_farsi import PersianPhonemizer

    class Encoded(dict):
        def to(self, device):
            return self

    phonemizer = PersianPhonemizer.__new__(PersianPhonemizer)
    phonemizer.device = "cpu"
    phonemizer.tokenizer = lambda *args, **kwargs: Encoded(input_ids=torch.zeros((1, input_length)))
    phonemizer.model = SimpleNamespace(generate=lambda **kwargs: torch.zeros((1, output_length)))
    with pytest.raises(ValueError, match="limit"):
        phonemizer("سلام")


def test_farsi_glottal_stop_and_zh_are_passed_verbatim():
    handler, calls = _farsi_handler()
    handler.phonemizer = lambda text: "?eqtesAde1 ?AmrikA ;Ale"
    list(handler._generate_audio("اقتصاد آمریکا", None))
    assert [call[1] for call in calls] == ["?eqtesAde ?AmrikA ;Ale"]
    assert calls[0][2] == {"frames_after_eos": 0, "copy_state": True}


def test_farsi_sentence_without_space_and_decimal():
    handler, _ = _farsi_handler()
    converted = []
    handler.phonemizer = lambda text: converted.append(text) or "salAm"
    list(handler._generate_audio("سلام.عدد ۳٫۵؟", None))
    assert converted == ["سلام", "عدد سه ممیز پنج دهم"]


@pytest.mark.parametrize("text", ["a1 b1 c1 d", "a1", "a11 b", "1a b"])
def test_farsi_rejects_invalid_or_oversized_ezafe_chain(text):
    handler, _ = _farsi_handler()
    with pytest.raises(ValueError):
        list(handler._phoneme_chunks(text))


def test_farsi_rejects_unknown_token():
    handler, _ = _farsi_handler()
    handler.model.flow_lm.conditioner.tokenizer.sp.encode = lambda text: [-1]
    with pytest.raises(ValueError, match="vocabulary"):
        list(handler._phoneme_chunks("salAm"))


@pytest.mark.parametrize("failure", ["silence", "length_cap", "nonfinite", "empty"])
def test_farsi_discards_bad_chunk_and_retries(failure):
    import torch

    handler, _ = _farsi_handler()
    bad = {
        "silence": [torch.zeros(2400)],
        "length_cap": [torch.ones(72000) * 0.1],
        "nonfinite": [torch.full((2400,), float("nan"))],
        "empty": [],
    }[failure]
    attempts = []

    def stream(**kwargs):
        attempts.append(kwargs)
        return iter(bad if len(attempts) == 1 else [torch.ones(2400) * 0.1])

    handler.model._generate_audio_stream_short_text = stream
    result = list(handler._synthesize_farsi_chunk("salAm", None))
    assert len(attempts) == 2
    assert len(result) == 1
    assert torch.allclose(result[0], torch.ones(2400) * 0.1)


def test_farsi_reports_two_failed_generations():
    import torch

    handler, _ = _farsi_handler()
    handler.model._generate_audio_stream_short_text = lambda **kwargs: iter([torch.zeros(2400)])
    with pytest.raises(RuntimeError, match="twice"):
        list(handler._synthesize_farsi_chunk("salAm", None))


def test_farsi_cancels_buffered_generation_without_emitting_audio():
    import torch

    handler, _ = _farsi_handler()
    stale = {"value": False}
    handler.cancel_scope = SimpleNamespace(is_stale=lambda generation: stale["value"])

    def stream(**kwargs):
        yield torch.ones(2400) * 0.1
        stale["value"] = True
        yield torch.ones(2400) * 0.1

    handler.model._generate_audio_stream_short_text = stream
    assert list(handler._synthesize_farsi_chunk("salAm", 0)) == []


def test_pocket_pcm_conversion_saturates_instead_of_wrapping():
    import numpy as np
    import torch

    from speech_to_speech.pipeline.messages import TTSInput

    handler, _ = _farsi_handler()
    handler.sample_rate = handler.model.sample_rate
    handler.phonemizer = None
    handler.model.generate_audio_stream = lambda *args, **kwargs: iter([torch.tensor([2.0, -2.0, 1.0, -1.0])])
    output = list(handler.process(TTSInput(text="Test")))
    assert np.array_equal(output[0][:4], np.array([32767, -32767, 32767, -32767], dtype=np.int16))


@pytest.mark.parametrize("output_rate", [16000, 24000])
def test_pocket_resampling_matches_whole_signal_across_chunk_boundaries(output_rate):
    import numpy as np
    import torch
    from scipy.signal import resample_poly

    from speech_to_speech.pipeline.messages import TTSInput

    handler, _ = _farsi_handler()
    handler.phonemizer = None
    handler.sample_rate = output_rate
    samples = (0.7 * np.sin(2 * np.pi * 997 * np.arange(12001) / 24000)).astype(np.float32)
    native_pcm = (samples * 32767).astype("<i2")
    if output_rate == 16000:
        expected = np.clip(np.round(resample_poly(native_pcm.astype(float), 2, 3)), -32768, 32767).astype("<i2")
    else:
        expected = native_pcm
    for cuts in [[], [1, 17, 768, 1011, 9999]]:
        handler.model.generate_audio_stream = lambda *args, **kwargs: (
            torch.from_numpy(part) for part in np.split(samples, cuts)
        )
        actual = np.concatenate(list(handler.process(TTSInput(text="Test"))))
        assert np.array_equal(actual[: len(expected)], expected)
        assert np.all(actual[len(expected) :] == 0)
        assert len(actual) == ((len(expected) + handler.blocksize - 1) // handler.blocksize) * handler.blocksize
