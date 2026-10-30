"""Opt-in native vLLM chat transport for Nemotron 3.5 Transcribe."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from speech_to_speech.language_codes import LANGUAGE_NAMES
from speech_to_speech.STT.openai_compatible_handler import (
    HttpTranscriptionOperation,
    HttpTranscriptionResult,
    TranscriptionRequestError,
)

TRANSCRIPTION_INSTRUCTION = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words "
    "and verbal fillers. Write numbers in their standard written form. Use <ev:*> and "
    "<sty:*> special tokens to recognize all non-speech verbatims and speaking style."
)
# Explicit vocabulary from the Nemotron 3.5 Transcribe tokenizer. Do not strip
# arbitrary angle-bracketed text, or blacklist real fillers such as "Uh-huh".
_EVENTS = (
    "breath",
    "crying",
    "effort_strain",
    "gesture_noise",
    "laughter",
    "mouth_noise",
    "music_vocal",
    "pause_timing",
    "shout_exclaim",
    "throat_cough",
)
_STYLES = (
    "anger_contempt",
    "authority_assertion",
    "calm_reflective",
    "fear_urgency",
    "positive_affect",
    "sadness_regret",
    "social_intent",
    "uncertainty_surprise",
    "vocal_intensity_pitch",
    "vocal_texture_delivery",
    "flat",
)
ANNOTATION_TOKENS = frozenset(
    [f"<spk:{i}>" for i in range(10)]
    + [f"<ev:{event}>" for event in _EVENTS]
    + [f"<sty:{boundary}:{style}>" for boundary in ("stt", "end") for style in _STYLES]
)
_ANNOTATIONS = re.compile("|".join(re.escape(token) for token in sorted(ANNOTATION_TOKENS)))


def clean_transcript(text: str) -> str:
    """Remove annotations, retaining lexical speech even when noise is present."""
    text = _ANNOTATIONS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:，。！？；：、])", r"\1", text)
    # Event-only output sometimes has trailing punctuation. An authoritative
    # empty final still reaches the turn lifecycle; it must not be dropped here.
    return text if any(character.isalnum() for character in text) else ""


def build_prompt(context: str, terms: tuple[str, ...], language: str | None) -> str:
    if any(not isinstance(term, str) or not term.strip() for term in terms):
        raise ValueError("STT context_terms must contain nonempty strings")
    parts = [TRANSCRIPTION_INSTRUCTION]
    if context:
        parts.append(context)
    # Deduplicate exact spellings while preserving configured order.
    unique_terms = list(dict.fromkeys(term.strip() for term in terms))
    if unique_terms:
        parts.append("Context terms: " + ", ".join(unique_terms) + ".")
        parts.append("Use these spellings only when supported by the audio.")
    if language and language.lower() != "auto":
        code = language.lower().replace("_", "-").split("-", 1)[0]
        name = LANGUAGE_NAMES.get(code)
        if name is None:
            raise ValueError("Nemotron STT language must be a recognized ISO language code")
        parts.append(f"Language: {name.title()}. Do not translate.")
    return " ".join(parts)


@dataclass
class NemotronTranscriptionOperation(HttpTranscriptionOperation):
    """Reuse HTTP cancellation, authentication, and errors for native audio chat."""

    prompt: str = TRANSCRIPTION_INSTRUCTION
    max_output_tokens: int = 384

    def _request_kwargs(self) -> dict[str, Any]:
        url = "data:audio/wav;base64," + base64.b64encode(self.wav_bytes).decode("ascii")
        return {
            "json": {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {"type": "audio_url", "audio_url": {"url": url}},
                        ],
                    }
                ],
                "temperature": 0,
                "seed": 0,
                "stream": False,
                "max_tokens": self.max_output_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": False,
            }
        }

    def _parse_response(self, body: bytes, content_type: str) -> HttpTranscriptionResult:
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptionRequestError("transcription server returned an invalid JSON response") from exc
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TranscriptionRequestError("transcription response must contain one choice")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            # A token-limit cutoff can contain repetitive noise hallucinations.
            # Surface a failed final instead of sending incomplete text to the LLM.
            raise TranscriptionRequestError("transcription response did not finish normally")
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TranscriptionRequestError("transcription response is missing string content")
        text = message["content"]
        if message.get("tool_calls") or message.get("refusal") or "<think>" in text or "</think>" in text:
            raise TranscriptionRequestError("transcription response contains unexpected assistant output")
        language = None if not self.language or self.language.lower() == "auto" else self.language
        return HttpTranscriptionResult(text=clean_transcript(text), language=language)
