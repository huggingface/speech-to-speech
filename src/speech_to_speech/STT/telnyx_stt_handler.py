"""Telnyx managed Speech-to-Text handler.

Streams each VAD segment to Telnyx's managed STT WebSocket API. One
WebSocket per utterance, matching the existing handler lifecycle.

Supports the full Telnyx engine catalog through one endpoint:
Telnyx (Whisper-based), Deepgram Nova-2/3/Flux, Google, Azure, xAI,
AssemblyAI, Speechmatics, Soniox, Parakeet.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.telnyx_ws import TelnyxSTTClient

logger = logging.getLogger(__name__)
console = Console()


class TelnyxSTTHandler(BaseSTTHandler):
    """Handles Speech-to-Text via Telnyx's managed WebSocket API."""

    def setup(
        self,
        should_listen: Any = None,
        api_key: str = "",
        engine: str = "Telnyx",
        language: str = "en",
        model: str = "",
        input_format: str = "wav",
        partial_results: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: Any = None,
        speculative_turns: Any = None,
        enable_live_transcription: bool = False,
        live_transcription_update_interval: float = 0.25,
    ) -> None:
        resolved_key = api_key or os.environ.get("TELNYX_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Telnyx STT requires an API key. Set --telnyx_stt_api_key or the TELNYX_API_KEY env var."
            )

        self.api_key = resolved_key
        self.engine = engine
        self.language = language
        self.model = model
        self.input_format = input_format
        self.partial_results = partial_results
        self.gen_kwargs = gen_kwargs or {}
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.enable_live_transcription = enable_live_transcription
        self.live_transcription_update_interval = live_transcription_update_interval
        self.sample_rate = 16000

        self._last_partial_emit_s: float = 0.0
        self._last_partial_text: str = ""

        logger.info(
            "Telnyx STT ready: engine=%s language=%s input_format=%s partial_results=%s",
            self.engine,
            self.language,
            self.input_format,
            self.partial_results,
        )

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        """Send the VAD segment to Telnyx STT and yield the final transcript."""
        audio = np.asarray(vad_audio.audio, dtype=np.int16)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)

        client = TelnyxSTTClient(
            api_key=self.api_key,
            engine=self.engine,
            language=self.language,
            input_format=self.input_format,
            partial_results=self.partial_results,
            model=self.model,
        )

        final_text = ""
        partials_emitted = 0
        try:
            client.connect()
            client.send_audio(audio, sample_rate=self.sample_rate)

            while True:
                event = client.recv_transcript()
                if event is None:
                    break

                event_type = event.get("type")
                if event_type == "error":
                    logger.error("Telnyx STT error: %s", event.get("error"))
                    break

                if event_type != "transcript":
                    continue

                text = event.get("transcript", "") or ""
                is_final = bool(event.get("is_final", False))

                if is_final:
                    final_text = text
                    break

                if self.enable_live_transcription and text:
                    partial = self._maybe_emit_partial(text, vad_audio)
                    if partial is not None:
                        partials_emitted += 1
                        yield partial
        except Exception as e:
            logger.error("Telnyx STT request failed: %s", e, exc_info=True)
        finally:
            client.close()

        if final_text.strip():
            console.print(f"[yellow]USER: {final_text.strip()}")

        yield Transcription(
            text=final_text,
            language_code=self.language,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    def _maybe_emit_partial(self, text: str, vad_audio: STTIn) -> PartialTranscription | None:
        """Build a rate-limited partial transcription for live display.

        Returns ``None`` when the update should be suppressed (duplicate text
        or within the throttle window).
        """
        now = perf_counter()
        if text == self._last_partial_text:
            return None
        if now - self._last_partial_emit_s < self.live_transcription_update_interval:
            return None
        self._last_partial_text = text
        self._last_partial_emit_s = now
        return PartialTranscription(
            text=text,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
        )
