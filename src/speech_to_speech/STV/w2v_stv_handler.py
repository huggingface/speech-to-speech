"""Wav2Vec2 phoneme-to-viseme extraction adapted from Fabio Catania's PR #99.

The original phoneme mapping is retained. Audio and response lifecycle messages
use the current typed pipeline instead of the PR's pickle socket protocol.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from importlib.resources import files
from typing import Any

import numpy as np

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.events import PipelineEvent
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, AudioOutput
from speech_to_speech.pipeline.queue_types import AudioOutItem
from speech_to_speech.pipeline.visemes import Viseme
from speech_to_speech.utils.utils import resolve_device

logger = logging.getLogger(__name__)


class Wav2Vec2STVHandler(BaseHandler[AudioOutItem, AudioOutItem]):
    """Buffer up to half a second for inference and preserve every audio sample.

    Listening and completion remain owned by the Realtime router. The stage
    never fabricates sentence boundaries or changes a TTS backend's contract.
    """

    SAMPLE_RATE = 16000
    MIN_AUDIO_LENGTH = 0.5
    BLOCKSIZE = 512

    def setup(
        self,
        model_name: str = "bookbot/wav2vec2-ljspeech-gruut",
        device: str = "auto",
        skip: bool = False,
        cancel_scope: CancelScope | None = None,
    ) -> None:
        self.skip = skip
        self.cancel_scope = cancel_scope
        self.asr_pipeline: Any = None
        self.phoneme_viseme_map: dict[str, list[int]] = {}
        self._reset()
        if skip:
            return

        # Importing Transformers and loading the model are opt-in.
        from transformers import pipeline

        self.device = resolve_device(device, ("cuda", "mps", "cpu"), "Speech-to-viseme")
        resource = files("speech_to_speech.STV").joinpath("phoneme_viseme_map.json")
        self.phoneme_viseme_map = json.loads(resource.read_text(encoding="utf-8"))
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name, device=self.device)
        if self.asr_pipeline.feature_extractor.sampling_rate != self.SAMPLE_RATE:
            raise ValueError("The viseme checkpoint must accept 16000 Hz audio.")
        self.warmup()

    def _reset(self) -> None:
        self._pending: list[bytes] = []
        self._pending_samples = 0
        self._sample_offset = 0
        self._context: tuple[str | None, int | None] | None = None

    def warmup(self) -> None:
        self.speech_to_visemes(np.zeros(int(self.MIN_AUDIO_LENGTH * self.SAMPLE_RATE), dtype=np.int16))

    def speech_to_visemes(self, audio: np.ndarray) -> list[Viseme]:
        """Map whole phoneme tokens, retaining the original PR's mouth shapes."""
        duration = len(audio) / self.SAMPLE_RATE
        waveform = audio.astype(np.float32) / 32768.0
        # Wav2Vec2's convolution needs a minimum input length. Padding is for
        # inference only, never for client playback or timestamp accounting.
        if len(waveform) < 400:
            waveform = np.pad(waveform, (0, 400 - len(waveform)))
        try:
            result = self.asr_pipeline(waveform, return_timestamps="char")
            cues = []
            for chunk in result.get("chunks", []):
                timestamp = chunk.get("timestamp")
                if not timestamp or timestamp[0] is None or timestamp[1] is None:
                    continue
                start, end = map(float, timestamp)
                if not np.isfinite(start) or not np.isfinite(end) or end <= start:
                    continue
                start, end = max(0.0, start), min(duration, end)
                if end <= start:
                    continue
                phoneme = chunk.get("text", "").replace("ˈ", "").replace("ˌ", "")
                shapes = self.phoneme_viseme_map.get(phoneme, [])
                # A compound phoneme maps to consecutive shapes, not multiple
                # mutually exclusive shapes at the same instant.
                for index, shape in enumerate(shapes):
                    step = (end - start) / len(shapes)
                    cues.append(Viseme(viseme=shape, start_s=start + index * step, end_s=start + (index + 1) * step))
            return sorted(cues, key=lambda cue: cue.start_s)
        except Exception:
            logger.exception("Viseme extraction failed; preserving assistant audio")
            return []

    def _stale(self, context: tuple[str | None, int | None] | None) -> bool:
        return bool(
            context is not None
            and context[1] is not None
            and self.cancel_scope
            and self.cancel_scope.is_stale(context[1])
        )

    def should_process_input(self, item: AudioOutItem) -> bool:
        # Completion markers must survive cancellation to release response state.
        if isinstance(item, AudioOutput) and isinstance(item.audio, bytes) and item.audio == AUDIO_RESPONSE_DONE:
            return True
        if not super().should_process_input(item):
            if self._context == (getattr(item, "response_key", None), getattr(item, "cancel_generation", None)):
                self._reset()
            return False
        return True

    def _flush(self) -> Iterator[AudioOutput]:
        if not self._pending:
            return
        pcm = b"".join(self._pending)
        context = self._context
        offset = self._sample_offset / self.SAMPLE_RATE
        self._pending = []
        self._sample_offset += self._pending_samples
        self._pending_samples = 0
        if self._stale(context):
            return
        cues = self.speech_to_visemes(np.frombuffer(pcm, dtype="<i2"))
        if self._stale(context):
            return
        cues = [cue.model_copy(update={"start_s": cue.start_s + offset, "end_s": cue.end_s + offset}) for cue in cues]
        response_key, generation = context or (None, None)
        for index in range(0, len(pcm), self.BLOCKSIZE * 2):
            yield AudioOutput(
                audio=pcm[index : index + self.BLOCKSIZE * 2],
                response_key=response_key,
                cancel_generation=generation,
                visemes=cues if index == 0 else [],
            )

    def process_pipeline_event(self, event: PipelineEvent) -> Iterator[AudioOutItem]:
        # Tail audio must not arrive after text/tool transitions or response.done.
        yield from self._flush()
        yield event

    def process(self, data: AudioOutItem) -> Iterator[AudioOutItem]:
        if not isinstance(data, (AudioOutput, bytes, np.ndarray)):
            raise TypeError(f"Unexpected viseme input: {type(data).__name__}")
        payload = data.audio if isinstance(data, AudioOutput) else data
        if self.skip:
            yield data if isinstance(data, (AudioOutput, bytes)) else AudioOutput(audio=data)
            return
        context = (data.response_key, data.cancel_generation) if isinstance(data, AudioOutput) else (None, None)
        if isinstance(payload, bytes) and payload == AUDIO_RESPONSE_DONE:
            if context == self._context and not (isinstance(data, AudioOutput) and data.cleanup_only):
                yield from self._flush()
                self._reset()
            elif context == self._context:
                self._reset()
            yield data
            return
        if self._stale(context):
            if context == self._context:
                self._reset()
            return
        if context != self._context:
            yield from self._flush()
            self._reset()
            self._context = context
        pcm = payload if isinstance(payload, bytes) else payload.tobytes()
        # Current TTS outputs are mono PCM16 at the pipeline's fixed rate.
        if len(pcm) % 2:
            raise ValueError("Viseme input must contain complete PCM16 samples.")
        batch_bytes = int(self.MIN_AUDIO_LENGTH * self.SAMPLE_RATE) * 2
        while pcm:
            count = min(len(pcm), batch_bytes - self._pending_samples * 2)
            self._pending.append(pcm[:count])
            self._pending_samples += count // 2
            pcm = pcm[count:]
            if self._pending_samples * 2 == batch_bytes:
                yield from self._flush()

    def on_session_end(self) -> None:
        self._reset()

    def cleanup(self) -> None:
        self._reset()
