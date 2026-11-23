"""Japanese synthesis with Irodori-TTS, a flow-matching model with caption-based voice design."""

from __future__ import annotations

import logging
from functools import partial
from math import gcd
from pathlib import Path
from threading import Event
from typing import Any, Iterator

import numpy as np
import scipy.signal

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.console import console
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.speech_languages import SpeechLanguageCapabilities
from speech_to_speech.TTS.base_tts_handler import BaseTTSHandler
from speech_to_speech.utils.utils import resolve_device

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000
IRODORI_DEVICES = ("cuda", "xpu", "mps", "cpu")

# With neither a caption nor a reference clip the model draws a new voice for every sentence.
# This caption keeps one voice for the whole conversation: "A young woman speaking calmly and clearly."
DEFAULT_CAPTION = "落ち着いた、聞き取りやすい声で話す若い女性。"
# A caption leaves some of the voice to the sampling noise, so every sentence starts from the same noise.
VOICE_SEED = 0


class IrodoriTTSHandler(BaseTTSHandler):
    def setup(
        self,
        should_listen: Event,
        model_name: str = "Aratako/Irodori-TTS-v4.1-Small-MF",
        device: str = "auto",
        caption: str | None = None,
        ref_audio: str | None = None,
        num_steps: int | None = None,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,  # Unused, for pipeline compatibility
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        if blocksize <= 0:
            raise ValueError(f"blocksize must be positive, got {blocksize}")
        if num_steps is not None and num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if ref_audio is not None and not Path(ref_audio).is_file():
            raise ValueError(f"Irodori reference audio not found: {ref_audio}")

        try:
            from irodori_tts.inference_runtime import (
                InferenceRuntime,
                RuntimeKey,
                SamplingRequest,
                download_hf_checkpoint,
            )
        except ImportError as exc:
            raise ImportError(
                "Irodori-TTS is not installed. It is not on PyPI; see the Irodori-TTS section of the README "
                "for the pinned install commands."
            ) from exc

        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.blocksize = blocksize
        self.language_capabilities = SpeechLanguageCapabilities(
            languages=frozenset({"ja"}), supports_language_selection=False, supports_auto=False
        )

        if caption is None and ref_audio is None:
            caption = DEFAULT_CAPTION
        device = resolve_device(device, IRODORI_DEVICES, "Irodori")
        checkpoint = model_name if Path(model_name).is_file() else download_hf_checkpoint(model_name)

        logger.info("Loading Irodori-TTS checkpoint %r on %s", model_name, device)
        self.runtime = InferenceRuntime.from_key(
            RuntimeKey(checkpoint=checkpoint, model_device=device, codec_device=device)
        )
        self._request = partial(
            SamplingRequest,
            caption=caption,
            ref_wav=ref_audio,
            no_ref=ref_audio is None,
            num_steps=num_steps,
            seed=VOICE_SEED,
        )
        self.warmup()

    def warmup(self) -> None:
        # The first call takes seconds: it encodes the reference clip, which the runtime then caches,
        # and pays the device's one-time startup cost. Spend that here, not on the first reply.
        logger.info("Warming up %s", self.__class__.__name__)
        self.runtime.synthesize(self._request(text="こんにちは。"))
        logger.info("%s warmed up", self.__class__.__name__)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            if self._gate_end_of_response(tts_input):
                yield AUDIO_RESPONSE_DONE
            return

        if not self._admit_input(tts_input):
            return

        text = tts_input.text
        if not text.strip():
            return

        cancel_generation = self.cancel_scope.generation if self.cancel_scope else None

        console.print(f"[green]ASSISTANT: {text}")
        # Synthesis start: after the speculative gate, before provider work.
        timer = self._latency_timer(tts_input, cancel_generation)
        result = self.runtime.synthesize(self._request(text=text))

        # Sampling and decoding are blocking. If the user interrupted while they ran,
        # discard the completed buffer before it reaches the output client.
        if self._is_stale(cancel_generation):
            logger.info("Irodori TTS output cancelled (interruption)")
            return

        audio = result.audio.detach().float().cpu().numpy().reshape(-1)
        if audio.size == 0:
            return

        # Full waveform is available before resample / block assembly.
        timer.first_provider_audio()

        source_sample_rate = int(result.sample_rate)
        if source_sample_rate != PIPELINE_SAMPLE_RATE:
            divisor = gcd(source_sample_rate, PIPELINE_SAMPLE_RATE)
            audio = scipy.signal.resample_poly(
                audio,
                PIPELINE_SAMPLE_RATE // divisor,
                source_sample_rate // divisor,
            )

        audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

        def emit(block: np.ndarray) -> Iterator[TTSOut]:
            timer.first_yield()
            yield block

        full_block_samples = (len(audio_int16) // self.blocksize) * self.blocksize
        for offset in range(0, full_block_samples, self.blocksize):
            if self._is_stale(cancel_generation):
                logger.info("Irodori TTS output cancelled (interruption)")
                return
            yield from emit(audio_int16[offset : offset + self.blocksize])

        if full_block_samples < len(audio_int16):
            if self._is_stale(cancel_generation):
                logger.info("Irodori TTS output cancelled (interruption)")
                return
            tail = audio_int16[full_block_samples:]
            yield from emit(np.pad(tail, (0, self.blocksize - len(tail))))
