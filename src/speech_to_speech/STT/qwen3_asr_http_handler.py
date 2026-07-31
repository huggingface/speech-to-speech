from __future__ import annotations

import base64
import io
import logging
import wave
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler
from speech_to_speech.STT.qwen3_asr_handler import CODE_BY_LANGUAGE_NAME

logger = logging.getLogger(__name__)
console = Console()

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _encode_wav_data_uri(audio: np.ndarray, sample_rate: int = 16000) -> str:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _fix_char_repeats(text: str, threshold: int) -> str:
    # Qwen3-ASR occasionally degenerates into long repeated character runs; this mirrors the
    # char-level pass of qwen_asr.inference.utils.detect_and_fix_repetitions. Reimplemented here
    # (rather than imported) since this handler intentionally has no dependency on the qwen_asr
    # package or its transformers==4.57.6 pin — it only ever talks to qwen-asr-serve over HTTP.
    result = []
    i = 0
    n = len(text)
    while i < n:
        count = 1
        while i + count < n and text[i + count] == text[i]:
            count += 1
        if count > threshold:
            result.append(text[i])
        else:
            result.append(text[i : i + count])
        i += count
    return "".join(result)


def _parse_asr_output(raw: str) -> tuple[str, str]:
    """Parse qwen-asr-serve's raw text into (language_name, text).

    Mirrors qwen_asr.inference.utils.parse_asr_output's tagged-output cases:
    "language English<asr_text>hello" -> ("English", "hello"); untagged output is
    returned as plain text with an empty language.
    """
    if not raw:
        return "", ""
    text = _fix_char_repeats(raw.strip(), threshold=20)
    if _ASR_TEXT_TAG not in text:
        return "", text.strip()

    meta_part, text_part = text.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta_part.lower():
        return "", text_part.strip()

    language = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if line.lower().startswith(_LANG_PREFIX):
            value = line[len(_LANG_PREFIX) :].strip()
            if value:
                language = value[:1].upper() + value[1:].lower()
            break
    return language, text_part.strip()


class Qwen3ASRHTTPSTTHandler(BaseSTTHandler):
    """
    Transcribes via a separately-running `qwen-asr-serve` HTTP server rather than loading
    Qwen3-ASR in-process. See Qwen3ASRSTTHandler / STT/README.md for why: qwen-asr==0.0.6
    hard-pins transformers==4.57.6, which conflicts with this project's transformers==5.6.2
    pin on macOS. Running the model as its own process (own virtualenv, own transformers
    version) sidesteps the conflict entirely instead of patching around it.
    """

    def setup(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_s: float = 30.0,
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}: checking {self.base_url}")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            self._transcribe(dummy_audio)
        except Exception:
            logger.exception(
                "%s: could not reach qwen-asr-serve at %s. Is it running? e.g. in a separate "
                "virtualenv pinned to transformers==4.57.6: "
                "`qwen-asr-serve Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8000`",
                self.__class__.__name__,
                self.base_url,
            )
            raise

    def _transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        data_uri = _encode_wav_data_uri(audio)
        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "audio_url", "audio_url": {"url": data_uri}}],
                    }
                ]
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_asr_output(content)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering Qwen3-ASR (HTTP)...")

        language, pred_text = self._transcribe(vad_audio.audio)
        language_code = CODE_BY_LANGUAGE_NAME.get(language.lower()) if language else None

        logger.debug("finished Qwen3-ASR (HTTP) inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Qwen3-ASR (HTTP): {language_code}")

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )
