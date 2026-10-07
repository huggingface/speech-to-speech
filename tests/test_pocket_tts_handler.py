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


def test_pocket_tts_rejects_unsupported_model():
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    with pytest.raises(ValueError, match="Unsupported"):
        handler.setup(Event(), model_name="/models/custom.yaml")


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
    handler._failed_responses = set()
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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("دمای هوا -۵ درجه است", "دمای هوا منفی پنج درجه است"),
        ("دمای هوا −۵ درجه است", "دمای هوا منفی پنج درجه است"),
        ("دمای هوا ۵ درجه است", "دمای هوا پنج درجه است"),
        ("-5", "منفی پنج"),
        ("−٥", "منفی پنج"),
        ("(-۳٫۵٪)", "منفی سه ممیز پنج دهم درصد"),
        ("−0.50", "منفی صفر ممیز پنج دهم"),
        ("-۵٫۰۰", "منفی پنج"),
        ("-۰۹۱۲", "منفی صفر نه یک دو"),
        ("−1234567890123", "منفی یک دو سه چهار پنج شش هفت هشت نه صفر یک دو سه"),
        ("۵-۱۰", "پنج ده"),
        ("نسخه-۵", "نسخه پنج"),
    ],
)
def test_persian_normalization_signed_numbers(text, expected):
    from speech_to_speech.TTS.persian_normalization import normalize

    assert normalize(text) == expected


def _run_farsi_inputs(handler, inputs):
    from queue import Queue

    from speech_to_speech.pipeline.messages import PIPELINE_END

    handler.stop_event = Event()
    handler.queue_in = Queue()
    handler.queue_out = Queue()
    handler.pipeline_index = None
    handler._times = []
    for item in inputs:
        handler.queue_in.put(item)
    handler.queue_in.put(PIPELINE_END)
    handler.run()
    return list(handler.queue_out.queue)


def _run_farsi_worker(handler, tts_input):
    from speech_to_speech.pipeline.messages import EndOfResponse

    terminal = EndOfResponse(
        turn_id=tts_input.turn_id,
        turn_revision=tts_input.turn_revision,
        cancel_generation=tts_input.cancel_generation,
        response_key=tts_input.response_key,
    )
    return _run_farsi_inputs(handler, [tts_input, terminal])


@pytest.mark.parametrize("failure_stage", ["silent_retries", "g2p", "phoneme_validation", "pcm_conversion"])
def test_farsi_worker_failure_finishes_realtime_response_as_failed(failure_stage):
    from queue import Queue

    import torch

    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.pipeline.events import ResponseFailedEvent
    from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput, TTSInput

    handler, calls = _farsi_handler()
    handler.phonemizer = lambda text: "salAm"
    if failure_stage == "silent_retries":

        def silent_stream(**kwargs):
            calls.append(kwargs)
            return iter([torch.zeros(2400)])

        handler.model._generate_audio_stream_short_text = silent_stream
    elif failure_stage == "g2p":

        def failed_g2p(text):
            raise RuntimeError("G2P failed")

        handler.phonemizer = failed_g2p
    elif failure_stage == "phoneme_validation":
        handler.phonemizer = lambda text: "invalid!"
    else:
        handler._generate_audio = lambda *args: iter([object()])
    tts_input = TTSInput(
        text="سلام", response_key="farsi-response", turn_id="turn-1", turn_revision=2, cancel_generation=7
    )
    outputs = _run_farsi_worker(handler, tts_input)

    assert len(outputs) == 3
    failure, completion, shutdown = outputs
    assert isinstance(failure, ResponseFailedEvent)
    assert (failure.response_key, failure.turn_id, failure.turn_revision, failure.cancel_generation) == (
        "farsi-response",
        "turn-1",
        2,
        7,
    )
    assert isinstance(completion, AudioOutput)
    assert completion.audio == AUDIO_RESPONSE_DONE
    assert completion.response_key == failure.response_key
    assert completion.cancel_generation == failure.cancel_generation
    assert shutdown == PIPELINE_END
    if failure_stage == "silent_retries":
        assert len(calls) == 2

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    conn_id = service.register()
    try:
        service.response._ensure_response(conn_id, tts_input.response_key)
        events = service.dispatch_pipeline_event(conn_id, failure)
        events.extend(service.finish_response(conn_id, response_key=completion.response_key))
        done = [event for event in events if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response.status == "failed"
        assert done[0].response.status_details.error.type == "response_failed"
        assert not any(event.type == "response.output_audio.delta" for event in events)
        assert not service._state(conn_id).in_response
    finally:
        service.unregister(conn_id)


def test_farsi_cancelled_exception_does_not_report_failure():
    from speech_to_speech.pipeline.cancel_scope import CancelScope
    from speech_to_speech.pipeline.events import ResponseFailedEvent
    from speech_to_speech.pipeline.messages import TTSInput

    handler, _ = _farsi_handler()
    handler.cancel_scope = CancelScope()

    def interrupted_g2p(text):
        handler.cancel_scope.cancel()
        raise RuntimeError("Interrupted G2P")

    handler.phonemizer = interrupted_g2p
    outputs = _run_farsi_worker(handler, TTSInput(text="سلام", response_key="cancelled", cancel_generation=0))
    assert not any(isinstance(output, ResponseFailedEvent) for output in outputs)
    assert not handler._failed_responses


def test_farsi_failed_response_skips_later_chunks_without_reopening_audio():
    from queue import Queue

    import torch

    from speech_to_speech.api.openai_realtime.service import RealtimeService
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor
    from speech_to_speech.pipeline.events import PipelineEvent, ResponseFailedEvent
    from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput, EndOfResponse, LLMResponseChunk

    handler, _ = _farsi_handler()
    conversions = []
    attempts = []

    def phonemizer(text):
        conversions.append(text)
        return "bad" if text == "خراب" else "salAm"

    def stream(**kwargs):
        phonemes = kwargs["text_to_generate"]
        attempts.append(phonemes)
        audio = torch.zeros(2400) if phonemes == "bad" else torch.full((2400,), 0.1)
        return iter([audio])

    handler.phonemizer = phonemizer
    handler.model._generate_audio_stream_short_text = stream
    processor = LMOutputProcessor.__new__(LMOutputProcessor)
    processor.setup()
    inputs = []
    for key, texts in [("failed", ["اول", "خراب", "آخر"]), ("next", ["بعدی"])]:
        identity = dict(response_key=key, turn_id=key, turn_revision=1, cancel_generation=0)
        for text in texts:
            inputs.extend(processor.process(LLMResponseChunk(text=text, **identity)))
        inputs.extend(processor.process(EndOfResponse(**identity)))
    outputs = _run_farsi_inputs(handler, inputs)
    assert conversions == ["اول", "خراب", "بعدی"]
    assert attempts == ["salAm", "bad", "bad", "salAm"]
    failures = [item for item in outputs if isinstance(item, ResponseFailedEvent)]
    assert len(failures) == 1
    assert failures[0].response_key == "failed"
    assert not handler._failed_responses

    service = RealtimeService(text_prompt_queue=Queue(), should_listen=Event())
    conn_id = service.register()
    events = []
    try:
        for item in outputs:
            if isinstance(item, PipelineEvent):
                events.extend(service.dispatch_pipeline_event(conn_id, item))
            elif isinstance(item, AudioOutput):
                if isinstance(item.audio, bytes) and item.audio == AUDIO_RESPONSE_DONE:
                    events.extend(service.finish_response(conn_id, response_key=item.response_key))
                else:
                    events.extend(service.encode_audio_chunk(conn_id, item.audio.tobytes(), item.response_key))
        terminals = [event for event in events if event.type == "response.done"]
        assert [event.response.status for event in terminals] == ["failed", "completed"]
        for terminal in terminals:
            audio = [
                event
                for event in events
                if event.type.startswith("response.output_audio.") and event.response_id == terminal.response.id
            ]
            assert any(event.type == "response.output_audio.delta" for event in audio)
            assert sum(event.type == "response.output_audio.done" for event in audio) == 1
            assert audio[-1].type == "response.output_audio.done"
    finally:
        service.unregister(conn_id)


@pytest.mark.parametrize("cleanup", ["terminal", "session"])
def test_farsi_failed_response_tracking_clears_on_matching_terminal_or_session_end(cleanup):
    from queue import Queue

    from speech_to_speech.pipeline.control import SESSION_END
    from speech_to_speech.pipeline.messages import EndOfResponse, TTSInput

    handler, calls = _farsi_handler()
    handler.queue_out = Queue()

    def failed_g2p(text):
        raise ValueError("G2P failed")

    handler.phonemizer = failed_g2p
    identity = dict(response_key="failed", turn_id="turn-1", turn_revision=1, cancel_generation=0)
    tts_input = TTSInput(text="سلام", **identity)
    assert list(handler.process(tts_input)) == []
    handler.phonemizer = lambda text: "salAm"
    # An unrelated terminal must not release this failed response.
    list(handler.process(EndOfResponse(response_key="other", cancel_generation=0)))
    assert list(handler.process(tts_input)) == []
    assert not calls
    terminal = EndOfResponse(**identity)
    reset = terminal if cleanup == "terminal" else SESSION_END
    outputs = _run_farsi_inputs(handler, [reset, tts_input, terminal])
    assert len(calls) == 1
    assert any(hasattr(output, "audio") and hasattr(output.audio, "dtype") for output in outputs)
    assert not handler._failed_responses


@pytest.mark.parametrize(
    "other_identity",
    [
        {"response_key": "other"},
        {"turn_id": "turn-2"},
        {"turn_revision": 2},
        {"cancel_generation": 1},
    ],
)
def test_farsi_failure_does_not_block_a_different_response_identity(other_identity):
    from queue import Queue

    from speech_to_speech.pipeline.messages import TTSInput

    handler, calls = _farsi_handler()
    handler.queue_out = Queue()

    def failed_g2p(text):
        raise ValueError("G2P failed")

    identity = dict(response_key="failed", turn_id="turn-1", turn_revision=1, cancel_generation=0)
    handler.phonemizer = failed_g2p
    assert list(handler.process(TTSInput(text="سلام", **identity))) == []
    handler.phonemizer = lambda text: "salAm"
    assert list(handler.process(TTSInput(text="سلام", **(identity | other_identity))))
    assert len(calls) == 1
    assert list(handler.process(TTSInput(text="سلام", **identity))) == []
    assert len(calls) == 1
