"""Telnyx managed Text-to-Speech handler.

Streams each sentence batch from the LLM to Telnyx's managed TTS WebSocket
API. The protocol has no per-utterance boundary marker inside a session, so a
session covers exactly one synthesis request and the handler opens one
WebSocket per call to :meth:`process`.

Supports the full Telnyx voice catalog through one endpoint:
Telnyx NaturalHD, AWS Polly, Azure, ElevenLabs, MiniMax, ResembleAI,
Inworld, Rime.
"""

from __future__ import annotations

import logging
import os
from threading import Event
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.utils.telnyx_ws import Mp3StreamDecoder, TelnyxTTSClient

logger = logging.getLogger(__name__)
console = Console()


class TelnyxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Handles Text-to-Speech via Telnyx's managed WebSocket API."""

    def setup(
        self,
        should_listen: Event,
        api_key: str = "",
        voice: str = "Telnyx.NaturalHD.astra",
        sample_rate: int = 16000,
        blocksize: int = 512,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("TELNYX_API_KEY", "")
        if not resolved_key:
            raise ValueError("Telnyx TTS requires an API key. Set --telnyx_tts_api_key or the TELNYX_API_KEY env var.")

        self.should_listen = should_listen
        self.api_key = resolved_key
        self.voice = voice
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns

        logger.info(
            "Telnyx TTS ready: voice=%s sample_rate=%d blocksize=%d",
            self.voice,
            self.sample_rate,
            self.blocksize,
        )

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

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text
        console.print(f"[green]ASSISTANT: {text}")

        client = TelnyxTTSClient(api_key=self.api_key, voice=self.voice)
        decoder = Mp3StreamDecoder(sample_rate=self.sample_rate)
        audio_buffer = np.array([], dtype=np.int16)
        try:
            client.connect()
            client.send_init()
            client.send_text(text)
            client.send_stop()

            while True:
                if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                    logger.info("TTS generation cancelled (interruption)")
                    return

                frame = client.recv_audio()
                if frame is None:
                    break

                mp3_bytes, is_final = frame
                if mp3_bytes:
                    audio_buffer = np.concatenate([audio_buffer, decoder.feed(mp3_bytes)])
                    while len(audio_buffer) >= self.blocksize:
                        yield audio_buffer[: self.blocksize]
                        audio_buffer = audio_buffer[self.blocksize :]
                if is_final:
                    break

            audio_buffer = np.concatenate([audio_buffer, decoder.flush()])
            while len(audio_buffer) >= self.blocksize:
                yield audio_buffer[: self.blocksize]
                audio_buffer = audio_buffer[self.blocksize :]
            if len(audio_buffer) > 0:
                yield np.pad(audio_buffer, (0, self.blocksize - len(audio_buffer)))
        except Exception as e:
            logger.error("Telnyx TTS request failed: %s", e, exc_info=True)
        finally:
            decoder.close()
            client.close()
