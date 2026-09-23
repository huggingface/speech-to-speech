from concurrent.futures import Future
from contextlib import contextmanager
from queue import Queue
from threading import Event, Thread

import numpy as np
import pytest
import torch

from speech_to_speech.diarization.streaming import SpeakerSegment
from speech_to_speech.diarization.worker import DiarizationWorker
from speech_to_speech.pipeline.events import SpeechStartedEvent, SpeechStoppedEvent, TranscriptionCompletedEvent
from speech_to_speech.pipeline.messages import Transcription, VADAudio
from speech_to_speech.pipeline.speaker_metadata import (
    PendingSpeakerAttribution,
    SpeakerAttribution,
    SpeakerInterval,
    SpeakerSession,
)
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.transcription_notifier import TranscriptionNotifier
from speech_to_speech.VAD.vad_handler import VADHandler


class FakeDiarizer:
    sample_rate = 16000

    def reset(self):
        self.processed_seconds = 0
        self.active_segments = ()
        self.samples = 0
        self.session_samples = 0

    def push(self, audio, *, sample_rate):
        start = self.samples / sample_rate
        speaker = 0 if self.session_samples < 3072 else 1
        self.samples += len(audio)
        self.session_samples += len(audio)
        self.processed_seconds = self.samples / sample_rate
        if audio.any():
            return [SpeakerSegment(speaker, start, self.processed_seconds)]
        return []

    def finish_utterance(self):
        self.samples = 0
        self.processed_seconds = 0
        return []


@contextmanager
def background(worker):
    thread = Thread(target=worker.run)
    thread.start()
    try:
        yield
    finally:
        worker.stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


def ready(attribution):
    future = Future()
    future.set_result(attribution)
    return PendingSpeakerAttribution(((future, SpeakerSession(), 0),))


def wait_pending(pending):
    for future, _, _ in pending.parts:
        future.result(timeout=2)


class FakeVAD:
    def reset_states(self):
        pass

    def __call__(self, audio, sampling_rate):
        return torch.tensor(float(audio.abs().max() > 0))


def make_vad(monkeypatch):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: (FakeVAD(), None))
    listen = Event()
    listen.set()
    handler = VADHandler(
        Event(),
        Queue(),
        Queue(),
        setup_args=(listen,),
        setup_kwargs=dict(
            speculative_turns=SpeculativeTurnTracker(),
            smart_turn=False,
            min_speech_ms=32,
            min_silence_ms=32,
            speech_pad_ms=0,
            speculative_reopen_ms=0,
            unanswered_reopen_ms=0,
            text_output_queue=Queue(),
        ),
    )
    handler.diarization_worker = DiarizationWorker(FakeDiarizer(), Event())
    return handler


def test_pcm_to_vad_to_stt_to_conversation_carries_distinct_speakers(
    monkeypatch, service, conn_id, runtime_config, text_prompt_queue
):
    vad = make_vad(monkeypatch)
    with background(vad.diarization_worker):
        outputs = []
        for loud in [True] * 4 + [False] * 3 + [True] * 4 + [False] * 2:
            outputs.extend(vad.process(np.full(512, 10000 if loud else 0, dtype=np.int16).tobytes()))
        assert len(outputs) == 2
        for output in outputs:
            wait_pending(output.speaker_pending)
        assert [{i.speaker for i in o.speaker_pending.resolve().intervals} for o in outputs] == [{0}, {1}]
        bridge = object.__new__(BaseSTTHandler)
        events = Queue()
        notifier = TranscriptionNotifier(Event(), Queue(), Queue(), setup_kwargs={"text_output_queue": events})
        for audio, text in zip(outputs, ["I would like tea.", "I would like coffee."]):
            service.dispatch_pipeline_event(conn_id, SpeechStartedEvent(turn_id=audio.turn_id, turn_revision=0))
            service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(turn_id=audio.turn_id, turn_revision=0))
            result = bridge.output_for_queue(Transcription(text=text, turn_id=audio.turn_id, turn_revision=0), audio)
            list(notifier.process(result))
            wire = service.dispatch_pipeline_event(conn_id, events.get_nowait())
            assert wire[0].transcript == text  # Metadata is not presented as recognized speech.
            request = text_prompt_queue.get_nowait()
            assert request.runtime_config is runtime_config
        messages = [m for m in runtime_config.chat.to_transformers_chat() if m["role"] == "user"]
        assert "speaker_0" in messages[0]["content"]
        assert "speaker_1" not in messages[0]["content"]
        assert "speaker_1" in messages[1]["content"]
        assert "I would like coffee." in messages[1]["content"]
        responses = str(runtime_config.chat.to_responses_api_chat())
        assert "speaker_0" in responses and "speaker_1" in responses
        # VAD reset cannot erase attribution from queued transcripts.
        snapshot = result.speaker_attribution.model_dump()
        vad.on_session_end()
        assert result.speaker_attribution.model_dump() == snapshot
        assert not outputs[0].speaker_pending.resolve().available


def test_reopened_audio_offsets_prefix_without_inventing_gap_speech(monkeypatch):
    vad = make_vad(monkeypatch)
    vad._total_samples = 32000
    vad._speculative_audio_prefix = np.zeros(16000)
    prefix = SpeakerAttribution(intervals=[SpeakerInterval(speaker=0, start=0, end=0.5)])
    vad._speculative_speaker_prefix = ready(prefix)
    current = ready(SpeakerAttribution(intervals=[SpeakerInterval(speaker=1, start=0, end=0.25)]))
    monkeypatch.setattr(vad.diarization_worker, "finalize", lambda start, end: current)
    result = vad._speaker_pending(4000).resolve()
    assert [(i.speaker, i.start, i.end) for i in result.intervals] == [(0, 0, 0.5), (1, 1, 1.25)]
    assert prefix.intervals[0].end == 0.5
    vad._start_new_turn()
    assert vad._speculative_speaker_prefix is None


def test_mixed_unknown_and_partial_audio_are_explicit():
    attribution = SpeakerAttribution(
        intervals=[SpeakerInterval(speaker=0, start=0, end=1), SpeakerInterval(speaker=1, start=0.5, end=1.5)],
        complete=False,
    )
    text = attribution.for_llm("hello")
    assert "individual words are not attributed" in text
    assert "additional speakers may be missing" in text
    assert "speaker_0" in text and "speaker_1" in text
    assert "unknown" in SpeakerAttribution(available=False).for_llm("hello")
    assert attribution.for_llm("") == ""


def test_revision_replaces_speaker_metadata_with_corrected_transcript(
    service, conn_id, runtime_config, text_prompt_queue
):
    for revision, speaker in [(0, 0), (1, 1)]:
        service.dispatch_pipeline_event(
            conn_id, SpeechStartedEvent(turn_id="turn_1", turn_revision=revision, reopened=bool(revision))
        )
        service.dispatch_pipeline_event(conn_id, SpeechStoppedEvent(turn_id="turn_1", turn_revision=revision))
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript=f"text {revision}",
                turn_id="turn_1",
                turn_revision=revision,
                speaker_attribution=SpeakerAttribution(intervals=[SpeakerInterval(speaker=speaker, start=0, end=1)]),
            ),
        )
    messages = [m for m in runtime_config.chat.to_transformers_chat() if m["role"] == "user"]
    assert len(messages) == 1
    assert "speaker_1" in messages[0]["content"] and "speaker_0" not in messages[0]["content"]
    assert messages[0]["content"].count("Speaker IDs are anonymous") == 1


def test_disabled_diarization_preserves_plain_transcript(service, conn_id, runtime_config):
    service.dispatch_pipeline_event(conn_id, TranscriptionCompletedEvent(transcript="plain text"))
    messages = [m for m in runtime_config.chat.to_transformers_chat() if m["role"] == "user"]
    assert messages[-1]["content"] == "plain text"
    source = VADAudio(audio=np.zeros(1))
    result = object.__new__(BaseSTTHandler).output_for_queue(Transcription(text="plain text"), source)
    assert result.speaker_attribution is None


def test_builder_loads_and_warms_separate_models_per_pipeline(monkeypatch):
    from types import SimpleNamespace

    from speech_to_speech import s2s_pipeline
    from speech_to_speech.diarization import StreamingDiarizer

    constructed = []
    warmed = []

    def load(model_id, **kwargs):
        model = FakeDiarizer()
        model.sample_rate = 16000
        model.warmup = lambda: warmed.append(model)
        constructed.append((model, model_id, kwargs))
        return model

    monkeypatch.setattr(StreamingDiarizer, "from_pretrained", load)
    monkeypatch.setattr(s2s_pipeline, "VADHandler", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(s2s_pipeline, "create_backend_handler", lambda *args: SimpleNamespace())
    args = s2s_pipeline.parse_arguments(
        ["--diarization_model_name", "fixture", "--diarization_revision", "preview", "--diarization_device", "mps"]
    )
    units = [
        s2s_pipeline._build_pipeline_unit(
            index=i,
            stop_event=Event(),
            module_kwargs=args.module_kwargs,
            vad_handler_kwargs=args.vad_handler_kwargs,
            stt_backend=args.stt_backend,
            llm_backend=args.llm_backend,
            tts_backend=args.tts_backend,
        )
        for i in range(2)
    ]
    assert len(constructed) == len(warmed) == 2
    assert units[0].handlers[0].diarization_worker is not units[1].handlers[0].diarization_worker
    assert constructed[0][2]["device"] == "mps"
    assert constructed[0][2]["revision"] == "preview"
    assert constructed[0][2]["streaming_mode"] == "low_latency"


def test_diarization_requires_stt_and_valid_threshold():
    from speech_to_speech.s2s_pipeline import parse_arguments, prepare_module_args

    for extra in [["--stt", "none"], ["--diarization_threshold", "0"]]:
        args = parse_arguments(["--diarization_model_name", "fixture", *extra])
        with pytest.raises(ValueError):
            prepare_module_args(args.module_kwargs, args.llm_backend)


def test_vad_idle_audio_never_reaches_diarization_and_preroll_is_sent_once(monkeypatch):
    vad = make_vad(monkeypatch)
    # Enable exact pre-roll and frequent progressive STT output.
    vad.iterator.speech_pad_samples = 512
    vad.enable_realtime_transcription = True
    vad.realtime_processing_pause = 0
    outputs = []
    for loud in [False] * 20 + [True] * 4 + [False] * 20:
        outputs.extend(vad.process(np.full(512, 10000 if loud else 0, dtype=np.int16).tobytes()))
    worker = vad.diarization_worker
    work = list(worker.queue.queue)  # No consumer: inspect what VAD enqueued.
    blocks = [item for item in work if item.audio is not None]
    # One pre-roll + four speech + two tail chunks, no long idle silence.
    assert sum(len(item.audio) for item in blocks) == 7 * 512
    assert blocks[0].start == 19 * 512
    assert all(a.start + len(a.audio) == b.start for a, b in zip(blocks, blocks[1:]))
    assert sum(item.result is not None for item in work) == 1
    assert any(output.mode == "progressive" for output in outputs)
    assert sum(output.mode == "final" for output in outputs) == 1


def test_unavailable_llm_metadata_is_compact_and_hides_invalid_labels():
    attribution = SpeakerAttribution(
        available=False,
        complete=True,
        intervals=[SpeakerInterval(speaker=2, start=0, end=1)],
    )
    assert attribution.for_llm("hello", include_explanation=False) == "[speaker=unknown, complete=false]\nhello"


def test_only_first_speaker_turn_has_explanation(service, conn_id, runtime_config):
    for turn in range(2):
        service.dispatch_pipeline_event(
            conn_id,
            TranscriptionCompletedEvent(
                transcript=f"hello {turn}",
                speaker_attribution=SpeakerAttribution(intervals=[SpeakerInterval(speaker=turn, start=0, end=1)]),
            ),
        )
    messages = [m["content"] for m in runtime_config.chat.to_transformers_chat() if m["role"] == "user"]
    assert "Speaker IDs are anonymous" in messages[0]
    assert "available" not in messages[1]
    assert messages[1] == "[speaker=speaker_1, complete=true]\nhello 1"
