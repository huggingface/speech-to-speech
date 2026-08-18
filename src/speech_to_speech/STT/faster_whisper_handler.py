from __future__ import annotations

import logging
import os
from typing import Any, Iterator

from faster_whisper import WhisperModel
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

console = Console()

logger = logging.getLogger(__name__)

# Whisper's own language set, which faster-whisper accepts verbatim and can report in
# TranscriptionInfo.language. Listed explicitly rather than derived from the LLM name map,
# so the STT language-coverage test actually compares two independent sources.
SUPPORTED_LANGUAGES = [
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
]  # fmt: skip


class FasterWhisperSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    # The language requested at setup; ``None`` means auto-detect. Declared here so an
    # instance built without setup() (as tests do) still resolves a language.
    start_language: Any = None

    def setup(
        self,
        model_name: str = "tiny.en",
        device: str = "auto",
        compute_type: str = "auto",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        # adapt_gen_kwargs() mutates and returns what it is given, so normalize a copy:
        # setup() must not consume the caller's configuration dict.
        gen_kwargs = self.adapt_gen_kwargs(dict(gen_kwargs))
        self.start_language = self._normalize_language(gen_kwargs.get("language"))
        if self.start_language is None:
            # faster-whisper auto-detects only when no language is passed at all.
            gen_kwargs.pop("language", None)
        self.gen_kwargs = gen_kwargs

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    @staticmethod
    def _normalize_language(language: Any) -> Any:
        """Return the language to transcribe with; ``None`` means auto-detect.

        Only ``None`` and a case-insensitive ``"auto"`` mean "detect". Anything else is
        passed through untouched so faster-whisper rejects an invalid language loudly
        rather than this silently falling back to auto-detection.
        """
        if isinstance(language, str) and language.strip().lower() == "auto":
            return None
        return language

    def _resolve_language(self, info: Any) -> Any:
        """The language code to report, based on what was actually transcribed.

        faster-whisper does not always honour the requested language: an English-only
        checkpoint (``tiny.en``, the default) overrides any request to ``en`` and records
        that in ``TranscriptionInfo.language``. Reporting the *request* would then tell the
        LLM and TTS to use a language the audio was never transcribed in, so always report
        what the model used. The request only decides whether this is auto-detect mode,
        which the ``-auto`` suffix signals to ``resolve_auto_language()``.
        """
        detected = getattr(info, "language", None)
        if not (isinstance(detected, str) and detected):
            # Nothing reported; the request is the best information available.
            return self.start_language

        if self.start_language is None:
            probability = getattr(info, "language_probability", None)
            if isinstance(probability, float):
                logger.debug("Faster Whisper detected language %s (p=%.2f)", detected, probability)
            return f"{detected}-auto"

        if detected != self.start_language:
            logger.warning(
                "Faster Whisper transcribed in %s despite the requested %s; reporting %s. "
                "An English-only checkpoint such as tiny.en always resolves to English.",
                detected,
                self.start_language,
                detected,
            )
        return detected

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering faster whisper...")

        segments, info = self.model.transcribe(vad_audio.audio, **self.gen_kwargs)
        output_text = []

        for segment in segments:
            logger.debug("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            output_text.append(segment.text)

        pred_text = " ".join(output_text).strip()

        logger.debug("finished whisper inference")
        if pred_text:
            console.print(f"[yellow]USER: {pred_text}")

            if vad_audio.mode == "progressive":
                yield PartialTranscription(
                    text=pred_text,
                    turn_id=vad_audio.turn_id,
                    turn_revision=vad_audio.turn_revision,
                )
                return

            yield Transcription(
                text=pred_text,
                language_code=self._resolve_language(info),
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
        else:
            logger.debug("no text detected. skipping...")

    def cleanup(self) -> None:
        logger.info("Stopping FasterWhisperSTTHandler")
        del self.model

    def adapt_gen_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        gen_kwargs["without_timestamps"] = not gen_kwargs.pop("return_timestamps", True)

        return gen_kwargs
