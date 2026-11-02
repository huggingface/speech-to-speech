"""Fish Audio Drama-3 through SGLang Omni's streaming speech endpoint."""

from __future__ import annotations

import base64
import os
from collections.abc import Callable, Iterator
from math import isfinite
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import soxr

from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.openai_compatible_handler import PIPELINE_SAMPLE_RATE, OpenAICompatibleTTSHandler


class Drama3TTSHandler(OpenAICompatibleTTSHandler):
    def setup(  # type: ignore[override]
        self,
        should_listen: Event,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        model: str = "fishaudio/drama-3",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        timeout: float = 300.0,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if bool(ref_audio) != bool(ref_text and ref_text.strip()):
            raise ValueError("Drama-3 voice cloning requires both --drama3_ref_audio and --drama3_ref_text.")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("Drama-3 timeout must be finite and > 0")
        self.references = []
        if ref_audio:
            self.references.append(
                {"data": base64.b64encode(Path(ref_audio).expanduser().read_bytes()).decode("ascii"), "text": ref_text}
            )
        super().setup(
            should_listen,
            base_url=base_url,
            # An unauthenticated local server must not receive OPENAI_API_KEY.
            api_key=api_key if api_key is not None else os.getenv("DRAMA3_API_KEY", ""),
            model=model,
            voice="default",
            sample_rate=44100,
            stream=True,
            timeout=timeout,
            blocksize=blocksize,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            gen_kwargs=gen_kwargs,
        )

    def _request_payload(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> dict[str, Any]:
        # The model infers language from text and uses references for its voice.
        # Named OpenAI voices and explicit language are not supported here.
        return {
            "temperature": 1.0,
            "top_p": 0.9,
            # Keep all semantic codes available, as in the official Drama-3 Space.
            "top_k": 4097,
            **self.gen_kwargs,
            "model": self.model,
            "input": text,
            "references": self.references,
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        }

    def _resample_to_blocks(
        self,
        sample_chunks: Iterator[np.ndarray],
        source_rate: int,
        on_first_source_audio: Callable[[], None] | None = None,
    ) -> Iterator[np.ndarray]:
        # The shared FIR expands 44.1 kHz audio 160-fold before filtering.
        # SoXR avoids that cost; LQ keeps its streaming filter delay short.
        resampler = soxr.ResampleStream(source_rate, PIPELINE_SAMPLE_RATE, 1, dtype="float64", quality="LQ")

        def converted_chunks() -> Iterator[np.ndarray]:
            nonlocal on_first_source_audio
            for samples in sample_chunks:
                if on_first_source_audio is not None and samples.size:
                    on_first_source_audio()
                    on_first_source_audio = None
                yield resampler.resample_chunk(samples.astype(np.float64), last=False)
            yield resampler.resample_chunk(np.empty(0, dtype=np.float64), last=True)

        yield from super()._resample_to_blocks(converted_chunks(), PIPELINE_SAMPLE_RATE)
