"""Regression coverage for the revived PR #99 without checkpoint downloads."""

from queue import Queue
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.control import SESSION_END
from speech_to_speech.pipeline.events import AssistantResponseDoneEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, PIPELINE_END, AudioOutput
from speech_to_speech.STV.w2v_stv_handler import Wav2Vec2STVHandler


def make_handler(monkeypatch, *, skip=False, inference=None, cancel_scope=None):
    calls = []

    def infer(audio, **kwargs):
        calls.append(audio.copy())
        if inference:
            return inference(audio)
        return {"chunks": [{"text": "tʃ", "timestamp": (0.0, min(0.2, len(audio) / 16000))}]}

    model = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=16000), __call__=infer)

    class FakeASR:
        feature_extractor = model.feature_extractor

        def __call__(self, *args, **kwargs):
            return infer(*args, **kwargs)

    loads = []

    def factory(*args, **kwargs):
        loads.append(kwargs)
        return FakeASR()

    # Resolve the lazy Transformers pipeline module before patching its export.
    import transformers.pipelines  # noqa: F401

    monkeypatch.setattr("transformers.pipeline", factory)
    handler = Wav2Vec2STVHandler(
        Event(), Queue(), Queue(), setup_kwargs={"skip": skip, "device": "cpu", "cancel_scope": cancel_scope}
    )
    calls.clear()  # Warmup is separate from the streamed audio assertions.
    return handler, calls, loads


def audio(samples, *, key="response-a", generation=0):
    return AudioOutput(audio=np.arange(samples, dtype=np.int16), response_key=key, cancel_generation=generation)


def done(*, key="response-a", generation=0, cleanup_only=False):
    return AudioOutput(
        audio=AUDIO_RESPONSE_DONE, response_key=key, cancel_generation=generation, cleanup_only=cleanup_only
    )


def pcm(items):
    return b"".join(item.audio if isinstance(item.audio, bytes) else item.audio.tobytes() for item in items)


@pytest.mark.parametrize("samples", [1, 399, 4096, 8000, 10240, 24001])
def test_completion_preserves_all_samples_including_short_tail(monkeypatch, samples):
    handler, _, _ = make_handler(monkeypatch)
    source = audio(samples)
    outputs = list(handler.process(source))
    terminal = done()
    outputs.extend(handler.process(terminal))
    assert outputs[-1] is terminal
    assert pcm(outputs[:-1]) == source.audio.tobytes()
    assert handler._pending_samples == 0
    assert all(item.response_key == source.response_key and item.cancel_generation == 0 for item in outputs)
    assert all(0 <= cue.start_s <= cue.end_s <= samples / 16000 for item in outputs[:-1] for cue in item.visemes)


def test_skip_does_not_load_or_warm_model(monkeypatch):
    handler, calls, loads = make_handler(monkeypatch, skip=True)
    source = audio(256)
    assert list(handler.process(source)) == [source]
    assert list(handler.process(done()))[0].audio == AUDIO_RESPONSE_DONE
    assert not calls and not loads


def test_batch_timestamps_continue_across_audio_segments(monkeypatch):
    handler, calls, _ = make_handler(monkeypatch)
    outputs = list(handler.process(audio(16000)))
    cues = [cue for item in outputs for cue in item.visemes]
    assert len(calls) == 2
    assert [cue.viseme for cue in cues] == [19, 16, 19, 16]
    assert [cue.start_s for cue in cues] == pytest.approx([0, 0.1, 0.5, 0.6])
    assert [cue.end_s for cue in cues] == pytest.approx([0.1, 0.2, 0.6, 0.7])


def test_model_receives_normalized_float_audio(monkeypatch):
    handler, calls, _ = make_handler(monkeypatch)
    samples = np.full(8000, 16384, dtype=np.int16)
    outputs = list(handler.process(AudioOutput(audio=samples)))
    assert calls[0].dtype == np.float32
    np.testing.assert_array_equal(calls[0], np.full(8000, 0.5, dtype=np.float32))
    assert pcm(outputs) == samples.tobytes()


def test_model_failure_preserves_audio_and_terminal(monkeypatch):
    def fail(_audio):
        raise RuntimeError("inference failed")

    handler, _, _ = make_handler(monkeypatch, inference=fail)
    source = audio(8000)
    outputs = list(handler.process(source))
    assert pcm(outputs) == source.audio.tobytes()
    assert not any(item.visemes for item in outputs)
    terminal = done()
    assert list(handler.process(terminal)) == [terminal]


def test_empty_reply_still_completes(monkeypatch):
    handler, calls, _ = make_handler(monkeypatch)
    terminal = done()
    assert list(handler.process(terminal)) == [terminal]
    assert not calls


def test_cancellation_drops_buffer_and_preserves_cleanup(monkeypatch):
    scope = CancelScope()
    handler, calls, _ = make_handler(monkeypatch, cancel_scope=scope)
    assert list(handler.process(audio(1024))) == []
    scope.cancel()
    terminal = done(cleanup_only=True)
    assert handler.should_process_input(terminal)
    assert list(handler.process(terminal)) == [terminal]
    assert not calls
    outputs = list(handler.process(audio(8000, key="response-b", generation=scope.generation)))
    assert outputs[0].visemes[0].start_s == 0
    assert all(item.response_key == "response-b" for item in outputs)


def test_cancellation_during_inference_drops_audio_and_visemes(monkeypatch):
    scope = CancelScope()
    handler, _, _ = make_handler(monkeypatch, cancel_scope=scope)

    def cancel(_audio, **_kwargs):
        scope.cancel()
        return {"chunks": [{"text": "a", "timestamp": (0, 0.2)}]}

    handler.asr_pipeline = cancel
    assert list(handler.process(audio(8000))) == []
    assert list(handler.process(done()))[0].audio == AUDIO_RESPONSE_DONE


def test_response_switch_flushes_tail_without_mixing_audio(monkeypatch):
    handler, _, _ = make_handler(monkeypatch)
    first = audio(1024)
    second = audio(8000, key="response-b")
    assert not list(handler.process(first))
    outputs = list(handler.process(second))
    previous = [item for item in outputs if item.response_key == "response-a"]
    current = [item for item in outputs if item.response_key == "response-b"]
    assert pcm(previous) == first.audio.tobytes()
    assert pcm(current) == second.audio.tobytes()
    assert current[0].visemes[0].start_s == 0


def test_ordered_event_flushes_tail_before_response_done(monkeypatch):
    handler, _, _ = make_handler(monkeypatch)
    source = audio(1024)
    event = AssistantResponseDoneEvent(response_key="response-a")
    handler.queue_in.put(source)
    handler.queue_in.put(event)
    handler.queue_in.put(done())
    handler.queue_in.put(PIPELINE_END)
    handler.run()
    outputs = list(handler.queue_out.queue)
    assert pcm(outputs[:2]) == source.audio.tobytes()
    assert outputs[2] is event
    assert outputs[3].audio == AUDIO_RESPONSE_DONE
    assert outputs[4] == PIPELINE_END


def test_session_reset_discards_pending_audio_and_resets_timeline(monkeypatch):
    handler, _, _ = make_handler(monkeypatch)
    source = audio(1024)
    handler.queue_in.put(source)
    handler.queue_in.put(SESSION_END)
    handler.queue_in.put(audio(8000, key="response-b"))
    handler.queue_in.put(PIPELINE_END)
    thread = Thread(target=handler.run)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    outputs = list(handler.queue_out.queue)
    assert outputs[0] is SESSION_END
    assert outputs[1].response_key == "response-b"
    assert outputs[1].visemes[0].start_s == 0
    assert pcm(outputs[1:-1]) == audio(8000, key="response-b").audio.tobytes()


def test_map_loaded_from_package_when_working_directory_changes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    handler, _, _ = make_handler(monkeypatch)
    assert handler.phoneme_viseme_map["tʃ"] == [19, 16]


def test_real_transformers_ctc_pipeline_supports_compound_phonemes(monkeypatch, tmp_path):
    """Exercise real model/tokenizer code with tiny local weights, without downloads."""
    import json

    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
    from transformers.pipelines import pipeline

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "tʃ": 4}
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(vocab))
    tokenizer = Wav2Vec2PhonemeCTCTokenizer(str(path), do_phonemize=False)
    config = Wav2Vec2Config(
        vocab_size=len(vocab),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8, 8, 8),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3),
        num_conv_pos_embeddings=8,
        num_conv_pos_embedding_groups=2,
        mask_time_prob=0,
    )
    model = Wav2Vec2ForCTC(config)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[4] = 10
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=Wav2Vec2FeatureExtractor(sampling_rate=16000),
        device="cpu",
    )
    handler, _, _ = make_handler(monkeypatch)
    handler.asr_pipeline = asr
    cues = handler.speech_to_visemes(np.ones(8000, dtype=np.int16))
    assert [cue.viseme for cue in cues] == [19, 16]
    assert 0 <= cues[0].start_s < cues[0].end_s == cues[1].start_s < cues[1].end_s <= 0.5


def test_stress_marks_do_not_hide_known_phonemes(monkeypatch):
    handler, _, _ = make_handler(
        monkeypatch,
        inference=lambda _audio: {
            "chunks": [{"text": "ˈoʊ", "timestamp": (0, 0.2)}, {"text": "ˌtʃ", "timestamp": (0.2, 0.4)}],
        },
    )
    outputs = list(handler.process(audio(8000)))
    assert [cue.viseme for item in outputs for cue in item.visemes] == [8, 4, 19, 16]
