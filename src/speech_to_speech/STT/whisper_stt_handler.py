from __future__ import annotations

import logging
import re
from typing import Any, Iterator, Optional

import numpy as np
import torch
from rich.console import Console
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
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

    def _code_from_language_tokens(self, token_ids: Any) -> Optional[str]:
        """Resolve the first Whisper language special token in ``token_ids`` to its code."""
        language_tokens = self._language_token_ids()
        if not language_tokens:
            return None

        try:
            flat = token_ids.flatten().tolist() if hasattr(token_ids, "flatten") else list(token_ids)
        except (IndexError, TypeError):
            return None

        for token_id in flat:
            try:
                code = language_tokens.get(int(token_id))
            except (TypeError, ValueError):
                continue
            if code is not None:
                return code
        return None

    def _detect_language(self, input_features: Any) -> tuple[Any, Optional[str]]:
        """Detect the spoken language up front, returning ``(encoder_outputs, code)``.

        ``generate()`` excludes the decoder input ids from its output, and the language token
        is one of them, so the detected language is never observable in the generated
        sequence on the ``transformers`` versions supported here. Whisper exposes
        ``detect_language()`` for exactly this, so ask it directly.

        The encoder output is computed once here and handed back so ``generate()`` can reuse
        it instead of re-encoding. Forcing the detected language also stops ``generate()``
        from running its own internal detection, so this is cheaper than the plain auto path
        rather than an extra cost.

        Both calls run under ``torch.no_grad()``. Calling the encoder directly does not
        inherit the no-grad context that ``generate()`` applies internally, so without this
        the encoder output would keep its autograd graph alive for the whole of decoding --
        wasted memory for a forward-only pipeline, and enough to risk OOM with a large
        Whisper checkpoint.
        """
        detect_language = getattr(self.model, "detect_language", None)
        if detect_language is None:
            return None, None

        try:
            with torch.no_grad():
                encoder_outputs = self.model.get_encoder()(input_features)
                token_ids = detect_language(encoder_outputs=encoder_outputs)
        except Exception as e:  # pragma: no cover - depends on the installed transformers
            logger.warning("Whisper language detection failed, falling back: %s", e)
            return None, None

        return encoder_outputs, self._code_from_language_tokens(token_ids)

    def _language_from_prefix(self, pred_ids: Any) -> Optional[str]:
        """Read a language token off the generated sequence, if this version emits one.

        Older ``transformers`` echoed the forced decoder prefix
        (``<|startoftranscript|><|de|><|transcribe|>``) in ``generate()`` output. Current
        versions strip it, so this is only a fallback for when ``detect_language()`` is
        unavailable. Scanning for a real language token rather than slicing a fixed position
        is what keeps a *text* token from being mistaken for a language.
        """
        language_tokens = self._language_token_ids()
        if not language_tokens:
            return None

        try:
            row = pred_ids[0]
            token_ids = row.tolist() if hasattr(row, "tolist") else list(row)
        except (IndexError, TypeError):
            return None

        return self._code_from_language_tokens(token_ids[:_LANGUAGE_TOKEN_SCAN_DEPTH])

    def _forced_language(self) -> Optional[str]:
        """The language explicitly requested for generation, if any."""
        forced = self.gen_kwargs.get("language")
        return forced if isinstance(forced, str) and forced and forced != "auto" else None

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering whisper...")

        input_features = self.prepare_model_inputs(vad_audio.audio)
        forced_language = self._forced_language()

        gen_kwargs: dict[str, Any] = dict(self.gen_kwargs)
        language_code = forced_language
        if forced_language is None:
            # Auto-detect mode: ask Whisper which language this is, then force it so the
            # transcription and the reported code cannot disagree.
            encoder_outputs, detected = self._detect_language(input_features)
            if encoder_outputs is not None:
                gen_kwargs["encoder_outputs"] = encoder_outputs
            if detected is not None:
                gen_kwargs["language"] = detected
                language_code = detected

        pred_ids = self.model.generate(input_features, **gen_kwargs)

        if language_code is None:
            # detect_language() was unavailable. Fall back to a prefix token if this version
            # emits one, then to the last known language, and only then to the default.
            language_code = self._language_from_prefix(pred_ids) or self.last_language or DEFAULT_LANGUAGE

        # Report whatever language was actually transcribed, even if it is outside
        # SUPPORTED_LANGUAGES -- discarding a correct transcription because its language is
        # not on a downstream allowlist is the bug this handler had. Only remember supported
        # languages, so an unsupported one never becomes the sticky fallback.
        if language_code in SUPPORTED_LANGUAGES:
            self.last_language = language_code
        else:
            logger.warning("Whisper detected unsupported language: %s", language_code)

        pred_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)[0]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
