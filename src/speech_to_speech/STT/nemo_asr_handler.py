from __future__ import annotations

import logging
import re
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.LLM.utils import WHISPER_LANGUAGE_TO_LLM_LANGUAGE
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.language_detection import (
    detect_language_from_text,
    warm_language_detector,
)
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.utils.utils import resolve_device

logger = logging.getLogger(__name__)
console = Console()

SAMPLE_RATE = 16000

SUPPORTED_LANGUAGES = ["en"]
TEXT_DETECTION_LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sk",
    "hu",
    "ro",
    "bg",
    "el",
    "hr",
    "sl",
    "sr",
    "da",
    "no",
    "sv",
    "fi",
    "et",
    "lv",
    "lt",
]
SUPPORTED_DEVICES = ("cuda", "npu", "cpu")
_NEMOTRON_LANG_TAG = re.compile(r"\s*<([A-Za-z]{2})(?:-[A-Za-z]{2})?>\s*$")


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    if isinstance(result, dict) and "text" in result:
        return str(result["text"])
    return str(result)


def _reported_language_code(start_language: str) -> str:
    requested = (start_language or "").strip() or "en"
    if requested.lower() == "auto":
        return "en"
    return requested


def _is_nemotron_multilingual(model_name: str) -> bool:
    return "nemotron-3.5" in model_name.lower()


def _split_lang_tag(text: str) -> tuple[str, str | None]:
    match = _NEMOTRON_LANG_TAG.search(text)
    if match is None:
        return text.strip(), None
    language_code = match.group(1).lower()
    # Nemotron uses Bokmål's "nb"; the pipeline uses Whisper's "no" for Norwegian.
    if language_code == "nb":
        language_code = "no"
    return text[: match.start()].strip(), language_code


def _warn_reported_language(start_language: str) -> None:
    requested = (start_language or "").strip() or "en"
    if requested.lower() == "auto":
        logger.warning(
            "NeMo Parakeet Unified does not detect language; %r is a request, not a code. Reporting %r instead.",
            start_language,
            "en",
        )
        return
    if requested not in WHISPER_LANGUAGE_TO_LLM_LANGUAGE:
        logger.warning(
            "Unknown language code %r for parakeet_unified_language; "
            "--enable_lang_prompt and some TTS backends may not resolve it.",
            requested,
        )


class NemoASRSTTHandler(BaseSTTHandler):
    """Speech to text with a NeMo ASR checkpoint through ASRModel.transcribe."""

    _detects_utterance_language = False
    _detect_language_from_text = False

    def setup(
        self,
        model_name: str,
        device: str = "auto",
        language: str = "en",
        gen_kwargs: dict | None = None,
        checkpoint_filename: str | None = None,
        checkpoint_revision: str | None = None,
        detect_language_from_text: bool = False,
    ) -> None:
        logger.info("Loading NeMo ASR STT model: %s", model_name)
        self.device = resolve_device(device, SUPPORTED_DEVICES, "NeMo ASR")
        self.start_language = (language or "").strip() or "en"
        self.model_name = model_name
        self._detects_utterance_language = _is_nemotron_multilingual(model_name)
        self._detect_language_from_text = detect_language_from_text
        if not self._detects_utterance_language and not detect_language_from_text:
            _warn_reported_language(self.start_language)
        self.language = _reported_language_code(self.start_language)
        self.last_language = self.language
        self.gen_kwargs = dict(gen_kwargs or {})
        if detect_language_from_text:
            self._language_detector = warm_language_detector(tuple(TEXT_DETECTION_LANGUAGES))

        from nemo.collections.asr.models import ASRModel

        if checkpoint_filename is None:
            self.model = ASRModel.from_pretrained(model_name=model_name)
        else:
            filename = checkpoint_filename.strip()
            if not filename:
                raise ValueError(
                    "checkpoint_filename is empty; pass a .nemo filename or omit the argument to use from_pretrained"
                )
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name, filename, revision=checkpoint_revision)
            self.model = ASRModel.restore_from(path)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.warmup()

    def warmup(self) -> None:
        logger.info("Warming up %s", self.__class__.__name__)
        try:
            self._transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        except Exception:
            logger.warning("%s: warmup failed", self.__class__.__name__, exc_info=True)

    def _transcribe(self, audio: np.ndarray) -> str:
        if self._detects_utterance_language:
            set_prompt = getattr(self.model, "set_inference_prompt", None)
            if callable(set_prompt):
                set_prompt("auto")
            try:
                results = self.model.transcribe([audio], target_lang="auto")
            except TypeError:
                results = self.model.transcribe([audio])
        else:
            results = self.model.transcribe([audio])
        if not results:
            return ""
        return _extract_text(results[0]).strip()

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        audio = np.asarray(vad_audio.audio, dtype=np.float32)
        text = self._transcribe(audio)
        language_code = self.language
        if self._detects_utterance_language:
            text, detected = _split_lang_tag(text)
            if detected is not None:
                language_code = detected
                self.last_language = detected
        elif self._detect_language_from_text and vad_audio.mode != "progressive":
            detected = detect_language_from_text(text, getattr(self, "_language_detector", None))
            if detected is not None:
                language_code = detected
                self.last_language = detected
            else:
                language_code = self.last_language or self.language
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
            return
        console.print(f"[yellow]USER: {text}")
        yield Transcription(
            text=text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.speech_end_at_s,
        )
