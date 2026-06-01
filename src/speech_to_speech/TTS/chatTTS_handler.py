from __future__ import annotations

import logging
from threading import Event
from typing import Any, Iterator

import ChatTTS
import librosa
import numpy as np
import torch
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

console = Console()


class ChatTTSHandler(BaseHandler[TTSIn, TTSOut]):
    def setup(
        self,
        should_listen: Event,
        device: str = "cuda",
        gen_kwargs: dict[str, Any] = {},  # Unused
        stream: bool = True,
        chunk_size: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        text = tts_input.text

        _cancel_gen = self.cancel_scope.generation if self.cancel_scope else None
        console.print(f"[green]ASSISTANT: {text}")
        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = time.time() - start  # Removing this line makes it fail more often. I'm looking into it.

        wavs_gen = self.model.infer(text, params_infer_code=self.params_infer_code, stream=self.stream)

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                if (
                    _cancel_gen is not None
                    and self.cancel_scope is not None
                    and self.cancel_scope.is_stale(_cancel_gen)
                ):
                    logger.info("TTS generation cancelled (interruption)")
                    return
                if gen[0] is None or len(gen[0]) == 0:
                    return
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)[0]
                while len(audio_chunk) > self.chunk_size:
                    yield audio_chunk[: self.chunk_size]  # Return the first chunk_size samples of the audio data
                    audio_chunk = audio_chunk[self.chunk_size :]  # Remove the samples that have already been returned
                yield np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.chunk_size):
                yield np.pad(
                    audio_chunk[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                )
