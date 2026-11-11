"""NVIDIA Speech NIM (Riva) HTTP adapter using the pipeline's cancellable HTTP transport."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from math import isfinite
from threading import Event
from typing import Any

import httpx

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker
from speech_to_speech.TTS.openai_compatible_handler import (
    HttpSpeechOperation,
    OpenAICompatibleTTSHandler,
    SpeechRequestError,
)

logger = logging.getLogger(__name__)

# NIM limits apply after text normalisation, which can expand the input, so
# these leave headroom below 2,000 (Magpie) and 500 (Chatterbox) characters.
NIM_SPLIT_CHARS = 1800
CHATTERBOX_SPLIT_CHARS = 450
_ERROR_BODY_MAX_CHARS = 500
_NIM_VOICE_PATTERN = re.compile(r"^[\w-]+\.[a-z]{2,3}-[a-z]{2,4}\.[\w-]+(\.[\w-]+)?$", re.IGNORECASE)
_CJK_SENTENCE_END = "。！？"
_SENTENCE_BOUNDARY = re.compile(rf"(?<=[.!?])\s+|(?<=[{_CJK_SENTENCE_END}])\s*")
_SSML = re.compile(r"^<speak\b[^>]*>(.*)</speak>$", re.DOTALL)
_SSML_TAG = re.compile(r"<[^>]*>")
_SSML_WRAPPER_CHARS = len("<speak></speak>")
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
_DATE_DAY_YEAR_COMMA = re.compile(
    rf"\b({_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?),\s*((?:19|20)\d{{2}})\b", re.IGNORECASE
)


def fix_date_commas(text: str) -> str:
    """Drop the day-year comma in 'Month D, YYYY', which some NIM normalisers speak as 'sil'."""
    return _DATE_DAY_YEAR_COMMA.sub(r"\1 \2", text)


def is_nim_voice(voice: str) -> bool:
    return voice.startswith("Magpie-") or bool(_NIM_VOICE_PATTERN.match(voice))


def split_text(text: str, max_chars: int = NIM_SPLIT_CHARS) -> list[str]:
    """Split on sentence boundaries into pieces of at most ``max_chars``."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars + 1)
            cut = cut if cut > 0 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].rstrip())
            sentence = sentence[cut:].lstrip()
        if not sentence:
            continue
        separator = "" if not current or current[-1] in _CJK_SENTENCE_END else " "
        if current and len(current) + len(separator) + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current}{separator}{sentence}"
    if current:
        pieces.append(current)
    return pieces


def is_ssml(text: str) -> bool:
    return bool(_SSML.match(text.strip()))


def split_ssml(text: str, max_chars: int = NIM_SPLIT_CHARS) -> list[str]:
    """Send SSML whole if it fits; otherwise split its text into separate ``<speak>`` documents.

    Inner tags are dropped from split pieces, since a tag cannot span two requests.
    Entities such as ``&amp;`` are kept as written.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    match = _SSML.match(text)
    if match is None:
        raise ValueError("not an SSML <speak> document")
    plain = " ".join(_SSML_TAG.sub(" ", match.group(1)).split())
    logger.warning(
        "NVIDIA NIM TTS split a long SSML document; pronunciation tags were removed to fit the %d-character limit",
        max_chars,
    )
    return [f"<speak>{piece}</speak>" for piece in split_text(plain, max_chars - _SSML_WRAPPER_CHARS)]


def is_zeroshot(name: str) -> bool:
    return "zeroshot" in name.lower()


def default_max_chars(*names: str | None) -> int:
    return CHATTERBOX_SPLIT_CHARS if any(name and "chatterbox" in name.lower() for name in names) else NIM_SPLIT_CHARS


class NvidiaNimSpeechOperation(HttpSpeechOperation):
    def _request_body_kwargs(self) -> dict[str, Any]:
        # The NIM expects multipart form fields, like `curl -F`.
        return {"files": {name: (None, str(value)) for name, value in self.payload.items()}}

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = " ".join((await response.aread()).decode("utf-8", errors="replace").split())
        if len(body) > _ERROR_BODY_MAX_CHARS:
            body = body[:_ERROR_BODY_MAX_CHARS] + "..."
        message = f"NVIDIA NIM TTS returned HTTP {response.status_code}"
        raise SpeechRequestError(f"{message}: {body}" if body else message)

    def _validate_content_type(self, response: httpx.Response) -> None:
        media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        if media_type.startswith("text/") or media_type == "application/json" or media_type.endswith("+json"):
            raise SpeechRequestError("NVIDIA NIM TTS returned a non-audio response")


class NvidiaNimTTSHandler(OpenAICompatibleTTSHandler):
    """Reuse remote TTS cancellation, response identity, and PCM/WAV block handling.

    Streaming mode posts to ``/v1/audio/synthesize_online`` and decodes raw
    16-bit LPCM; offline mode posts to ``/v1/audio/synthesize`` and parses WAV.

    Voice resolution: ``voice_override`` always wins. Otherwise a voice requested
    by a realtime client is used only when it looks like a NIM voice name
    (``Magpie-...`` or ``Model.LOCALE.Speaker[.Emotion]``). Clients built for the
    default Qwen3-TTS backend send names such as ``Aiden``, which the NIM rejects,
    so those fall back to the configured voice with one INFO log per session.
    """

    # Provider setup options differ; processing and transport hooks are shared.
    def setup(  # type: ignore[override]
        self,
        should_listen: Event,
        base_url: str = "http://localhost:9000",
        voice: str = "Magpie-Multilingual.EN-US.Aria",
        language: str = "en-US",
        sample_rate: int = 22050,
        stream: bool = True,
        timeout_s: float = 30.0,
        api_key: str | None = None,
        voice_override: str | None = None,
        fix_date_commas: bool = True,
        max_chars: int | None = None,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("NVIDIA NIM sample rate must be > 0")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("NVIDIA NIM timeout must be finite and > 0")
        if max_chars is not None and max_chars <= _SSML_WRAPPER_CHARS:
            raise ValueError(f"NVIDIA NIM max chars must be > {_SSML_WRAPPER_CHARS}")
        voice_override = voice_override.strip() if voice_override and voice_override.strip() else None
        if is_zeroshot(voice) or (voice_override and is_zeroshot(voice_override)):
            raise ValueError("Magpie Zeroshot needs an audio prompt, which this backend does not send yet")
        self._ignored_voice_logged = False
        self.fix_date_commas = fix_date_commas
        self.max_chars = max_chars or default_max_chars(voice, voice_override)
        # The parent stream flag describes a vLLM-specific request extension;
        # here it only selects the NIM endpoint in _make_operation below. The
        # parent warms up with its voice, so an override is passed in for that.
        super().setup(
            should_listen,
            base_url=base_url,
            api_key=api_key or "",
            model="nvidia-nim",
            voice=voice_override or voice,
            language=language,
            response_format="pcm" if stream else "wav",
            sample_rate=sample_rate,
            stream=stream,
            timeout=timeout_s,
            blocksize=blocksize,
            cancel_scope=cancel_scope,
            speculative_turns=speculative_turns,
            gen_kwargs=gen_kwargs,
        )
        self.voice = voice
        self.voice_override = voice_override
        if voice_override:
            logger.info(
                "NVIDIA NIM TTS voice override active: %s (client-requested voices are ignored)", voice_override
            )
        self.endpoint_url = self._nim_endpoint_url()

    def _nim_endpoint_url(self) -> str:
        return f"{self.base_url}/v1/audio/{'synthesize_online' if self.stream else 'synthesize'}"

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield from super().process(tts_input)
            return
        if self.fix_date_commas:
            text = fix_date_commas(tts_input.text)
            if text != tts_input.text:
                tts_input = tts_input.model_copy(update={"text": text})
        pieces = self._split(tts_input.text)
        if len(pieces) <= 1:
            if pieces and pieces[0] != tts_input.text:
                tts_input = tts_input.model_copy(update={"text": pieces[0]})
            yield from super().process(tts_input)
            return
        # One generation for all pieces, so a cancel during any piece stops the rest.
        update: dict[str, Any] = {}
        if tts_input.cancel_generation is None and self.cancel_scope is not None:
            update["cancel_generation"] = self.cancel_scope.generation
        for piece in pieces:
            yield from super().process(tts_input.model_copy(update={**update, "text": piece}))

    def _split(self, text: str) -> list[str]:
        if is_ssml(text):
            return split_ssml(text, self.max_chars)
        if len(text.strip()) <= self.max_chars:
            return [text]
        return split_text(text, self.max_chars)

    def _request_payload(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> dict[str, Any]:
        # NIM locales such as en-US do not match pipeline language codes, so the
        # configured language is always sent.
        return {
            "text": text,
            "language": self.language,
            "voice": voice if isinstance(voice, str) else self.voice,
            "sample_rate_hz": self.sample_rate,
        }

    def _make_operation(
        self,
        *,
        text: str,
        voice: str | dict[str, str],
        language: str | None = None,
        use_setup_language: bool = True,
    ) -> HttpSpeechOperation:
        return NvidiaNimSpeechOperation(
            endpoint_url=self._nim_endpoint_url(),
            api_key=self.api_key or None,
            payload=self._request_payload(text=text, voice=voice),
            timeout_s=self.timeout,
            response_format=self.response_format,
        )

    def _resolve_voice(self, runtime_config: RuntimeConfig | None, response: Any) -> str | dict[str, str]:
        if self.voice_override:
            return self.voice_override
        requested = super()._resolve_voice(runtime_config, response)
        if isinstance(requested, str) and is_nim_voice(requested):
            return requested
        if requested != self.voice and not self._ignored_voice_logged:
            logger.info(
                "NVIDIA NIM TTS ignoring client voice %r; using configured voice %s",
                requested if isinstance(requested, str) else requested.get("id"),
                self.voice,
            )
            self._ignored_voice_logged = True
        return self.voice

    def on_session_end(self) -> None:
        self._ignored_voice_logged = False
        super().on_session_end()
