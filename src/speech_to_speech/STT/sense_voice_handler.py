from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import empty_device_cache

logger = logging.getLogger(__name__)
console = Console()


class SenseVoiceSTTHandler(BaseSTTHandler):
    """Transcribe VAD segments with FunASR's SenseVoiceSmall checkpoint."""

    def setup(
        self,
        model_name: str = "FunAudioLLM/SenseVoiceSmall",
        device: str = "cuda",
        language: str = "auto",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SenseVoice STT requires the optional 'sensevoice' extra. "
                'Install it with `pip install "speech-to-speech[sensevoice]"`.'
            ) from exc

        self.device = device
        self.language = language
        self.gen_kwargs = dict(gen_kwargs or {})
        self._postprocess = rich_transcription_postprocess
        self.model = AutoModel(model=model_name, hub="hf", device=device, disable_update=True)
        self.warmup()

    def _generate(self, audio: np.ndarray) -> str:
        result = self.model.generate(
            audio,
            cache={},
            language=self.language,
            use_itn=True,
            **self.gen_kwargs,
        )
        return self._postprocess(result[0]["text"]).strip()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        self._generate(np.zeros(16000, dtype=np.float32))

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        pred_text = self._generate(vad_audio.audio)
        empty_device_cache(self.device)
        console.print(f"[yellow]USER: {pred_text}")

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.speech_end_at_s,
            )
