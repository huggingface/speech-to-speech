from __future__ import annotations

import logging
import re
from typing import Any, Iterator, Optional

import numpy as np
import torch
from rich.console import Console
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "es",
    "zh",
    "ja",
    "ko",
    "hi",
    "de",
    "pt",
    "pl",
    "it",
    "nl",
]

DEFAULT_LANGUAGE = "en"

# Whisper language special tokens look like "<|de|>" / "<|yue|>". No other Whisper
# special token ("<|transcribe|>", "<|nospeech|>", "<|notimestamps|>", ...) is 2-3
# letters long, so this pattern only ever matches language tokens.
_LANGUAGE_TOKEN_RE = re.compile(r"^<\|([a-z]{2,3})\|>$")

# The forced decoder prefix is "<|startoftranscript|><|xx|><|transcribe|><|notimestamps|>",
# so the language token is always within the first few ids when it is present at all.
_LANGUAGE_TOKEN_SCAN_DEPTH = 4


class WhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    _language_token_id_map: Optional[dict[int, str]] = None

    def setup(
        self,
        model_name: str = "distil-whisper/distil-large-v3",
        device: str = "cuda",
        torch_dtype: str = "float16",
        compile_mode: Optional[str] = None,
        language: Optional[str] = None,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.gen_kwargs = gen_kwargs
        self.start_language = language
        self.last_language = language if language != "auto" else None
        if self.last_language is not None:
            self.gen_kwargs["language"] = self.last_language

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)

        # compile
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(self.model.forward, mode=self.compile_mode, fullgraph=True)
        self.warmup()

    def prepare_model_inputs(self, audio: np.ndarray) -> torch.Tensor:
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device, dtype=self.torch_dtype)

        return input_features

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1 if self.compile_mode == "default" else 2
        dummy_input = torch.randn(
            (1, self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device,
        )
        if self.compile_mode not in (None, "default"):
            # generating more tokens than previously will trigger CUDA graphs capture
            # one should warmup with a number of generated tokens above max tokens targeted for subsequent generation
            # hence, having min_new_tokens < max_new_tokens in the future doesn't make sense
            warmup_gen_kwargs = {
                "min_new_tokens": self.gen_kwargs["max_new_tokens"],  # Yes, assign max_new_tokens to min_new_tokens
                "max_new_tokens": self.gen_kwargs["max_new_tokens"],
                **self.gen_kwargs,
            }
        else:
            warmup_gen_kwargs = self.gen_kwargs

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **warmup_gen_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()

            logger.info(
                f"{self.__class__.__name__}:  warmed up! time: {start_event.elapsed_time(end_event) * 1e-3:.3f} s"
            )

    def _language_token_ids(self) -> dict[int, str]:
        """Map the tokenizer's Whisper language special-token ids to their ISO codes."""
        if self._language_token_id_map is not None:
            return self._language_token_id_map

        tokenizer = self.processor.tokenizer
        mapping: dict[int, str] = {}
        for token in getattr(tokenizer, "all_special_tokens", None) or []:
            match = _LANGUAGE_TOKEN_RE.match(token)
            if match is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int):
                mapping[token_id] = match.group(1)

        self._language_token_id_map = mapping
        return mapping

    def _detected_language(self, pred_ids: Any) -> Optional[str]:
        """Read the language Whisper decoded, or ``None`` if it did not report one.

        The language token is only in the returned sequence when ``generate()`` echoes the
        forced decoder prefix. Recent ``transformers`` strips that prefix, so the leading ids
        are ordinary text. Scanning for a real language token instead of slicing a fixed
        position keeps this correct on both behaviours; guessing from position turns a text
        token into a bogus "language" and silently discards a good transcription.
        """
        language_tokens = self._language_token_ids()
        if not language_tokens:
            return None

        try:
            row = pred_ids[0]
            token_ids = row.tolist() if hasattr(row, "tolist") else list(row)
        except (IndexError, TypeError):
            return None

        for token_id in token_ids[:_LANGUAGE_TOKEN_SCAN_DEPTH]:
            try:
                code = language_tokens.get(int(token_id))
            except (TypeError, ValueError):
                continue
            if code is not None:
                return code
        return None

    def _forced_language(self) -> Optional[str]:
        """The language explicitly requested for generation, if any."""
        forced = self.gen_kwargs.get("language")
        return forced if isinstance(forced, str) and forced and forced != "auto" else None

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering whisper...")

        input_features = self.prepare_model_inputs(vad_audio.audio)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)

        forced_language = self._forced_language()
        detected_language = self._detected_language(pred_ids) or forced_language

        if detected_language in SUPPORTED_LANGUAGES:
            assert detected_language is not None
            language_code = detected_language
            self.last_language = language_code
        elif forced_language is not None:
            # The first pass already ran with the requested language, so it is authoritative.
            # Re-generating here would throw away a correct transcription.
            language_code = forced_language
        else:
            if detected_language is not None:
                logger.warning("Whisper detected unsupported language: %s", detected_language)
            last_language = self.last_language
            if last_language in SUPPORTED_LANGUAGES:
                assert last_language is not None
                # Auto-detection is unusable and we have a known-good language: retry with it.
                logger.debug("Reprocessing with the last known language: %s", last_language)
                pred_ids = self.model.generate(input_features, **{**self.gen_kwargs, "language": last_language})
                language_code = last_language
            else:
                # Nothing better to fall back to. Keep this pass rather than re-generating
                # with language=None, which would produce an identical result at double cost.
                language_code = detected_language or DEFAULT_LANGUAGE

        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
