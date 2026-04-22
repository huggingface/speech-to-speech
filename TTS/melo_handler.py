from __future__ import annotations

from melo.api import TTS
import logging
from baseHandler import BaseHandler
from cancel_scope import CancelScope
from pipeline_messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
import librosa
import numpy as np
from rich.console import Console
import torch

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-BR",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


class MeloTTSHandler(BaseHandler[TTSInput | EndOfResponse]):
    def setup(
        self,
        should_listen,
        device="mps",
        language="en",
        speaker_to_id="en",
        gen_kwargs={},  # Unused
        blocksize=512,
        cancel_scope: CancelScope | None = None,
    ):
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.device = device
        self.language = language
        self.model = TTS(
            language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[self.language], device=device
        )
        self.speaker_id = self.model.hps.data.spk2id[
            WHISPER_LANGUAGE_TO_MELO_SPEAKER[speaker_to_id]
        ]
        self.blocksize = blocksize
        self._initial_language = self.language
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("text", self.speaker_id, quiet=True)

    def process(self, tts_input: TTSInput | EndOfResponse):
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        gen = self.cancel_scope.generation if self.cancel_scope else None
        language_code = tts_input.language_code
        runtime_config = tts_input.runtime_config
        text = tts_input.text

        console.print(f"[green]ASSISTANT: {text}")

        if language_code is not None and self.language != language_code:
            try:
                self.model = TTS(
                    language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[language_code],
                    device=self.device,
                )
                self.speaker_id = self.model.hps.data.spk2id[
                    WHISPER_LANGUAGE_TO_MELO_SPEAKER[language_code]
                ]
                self.language = language_code
            except KeyError:
                console.print(
                    f"[red]Language {language_code} not supported by Melo. Using {self.language} instead."
                )

        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        try:
            audio_chunk = self.model.tts_to_file(
                text, self.speaker_id, quiet=True
            )
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Error in MeloTTSHandler: {e}")
            audio_chunk = np.array([])
        if len(audio_chunk) == 0:
            if not runtime_config:
                self.should_listen.set()
            return
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            if gen is not None and self.cancel_scope.is_stale(gen):
                logger.info("TTS generation cancelled (interruption)")
                return
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        if not runtime_config:
            self.should_listen.set()

    def on_session_end(self):
        if self.language != self._initial_language:
            self.language = self._initial_language
            self.model = TTS(
                language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[self.language], device=self.device
            )
            self.speaker_id = self.model.hps.data.spk2id[
                WHISPER_LANGUAGE_TO_MELO_SPEAKER[self.language]
            ]
        logger.debug("Melo TTS session state reset")
