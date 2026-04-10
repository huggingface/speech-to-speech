"""
MiniMax TTS Handler

Uses the MiniMax Text-to-Audio API (t2a_v2) to generate speech.
Requires MINIMAX_API_KEY environment variable.

API reference: https://platform.minimax.io/docs/api-reference/speech-t2a-http
"""

import json
import logging
import os
import urllib.request
from time import perf_counter

import numpy as np
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

PIPELINE_SR = 16000  # Pipeline expects 16 kHz int16 audio

MINIMAX_TTS_VOICES = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
]


class MiniMaxTTSHandler(BaseHandler):
    """
    Text-to-Speech handler using the MiniMax t2a_v2 API.

    Streams PCM audio via SSE, converts hex-encoded PCM chunks to int16
    numpy arrays, and feeds them into the pipeline at 16 kHz.

    Supported models:
        - speech-2.8-hd   (default, highest quality)
        - speech-2.8-turbo (faster)
    """

    def setup(
        self,
        should_listen,
        api_key=None,
        base_url="https://api.minimax.io",
        model="speech-2.8-hd",
        voice="English_Graceful_Lady",
        speed=1.0,
        vol=1.0,
        pitch=0,
        blocksize=512,
        gen_kwargs=None,
    ):
        self.should_listen = should_listen
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable "
                "or pass api_key to MiniMaxTTSHandler."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.speed = speed
        self.vol = vol
        self.pitch = pitch
        self.blocksize = blocksize

        logger.info(
            f"MiniMax TTS handler ready (model={self.model}, voice={self.voice})"
        )

    def _stream_pcm_chunks(self, text):
        """
        Call the MiniMax t2a_v2 API with stream=True and PCM format.
        Yields int16 numpy arrays as audio arrives via SSE.
        """
        url = f"{self.base_url}/v1/t2a_v2"
        payload = json.dumps(
            {
                "model": self.model,
                "text": text,
                "stream": True,
                "voice_setting": {
                    "voice_id": self.voice,
                    "speed": self.speed,
                    "vol": self.vol,
                    "pitch": self.pitch,
                },
                "audio_setting": {
                    "sample_rate": PIPELINE_SR,
                    "format": "pcm",
                    "channel": 1,
                },
                "stream_options": {
                    "exclude_aggregated_audio": True,
                },
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as response:
            buf = b""
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                buf += chunk
                # Process all complete lines
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str.startswith("data:"):
                        continue
                    json_str = line_str[5:].strip()
                    if not json_str or json_str == "[DONE]":
                        continue
                    try:
                        event = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    # Check for API-level errors
                    base_resp = event.get("base_resp", {})
                    if base_resp.get("status_code", 0) not in (0, None):
                        logger.error(
                            f"MiniMax TTS API error: {base_resp.get('status_msg')}"
                        )
                        return

                    status = event.get("data", {}).get("status")
                    if status == 1:
                        hex_audio = event.get("data", {}).get("audio", "")
                        if hex_audio:
                            try:
                                raw = bytes.fromhex(hex_audio)
                                yield np.frombuffer(raw, dtype=np.int16).copy()
                            except ValueError:
                                logger.warning("Failed to decode hex audio chunk")

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple) and llm_sentence[0] == "__END_OF_RESPONSE__":
            yield b"__RESPONSE_DONE__"
            return

        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence  # Unpack (text, language_code)

        if not llm_sentence or not llm_sentence.strip():
            return

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        start = perf_counter()
        first_chunk = True
        leftover = np.array([], dtype=np.int16)

        try:
            for audio_chunk in self._stream_pcm_chunks(llm_sentence):
                if first_chunk:
                    logger.debug(
                        f"MiniMax TTS TTFA: {perf_counter() - start:.3f}s"
                    )
                    first_chunk = False

                audio_chunk = np.concatenate([leftover, audio_chunk])

                # Yield exactly blocksize-sized chunks
                n = (len(audio_chunk) // self.blocksize) * self.blocksize
                for i in range(0, n, self.blocksize):
                    yield audio_chunk[i : i + self.blocksize]
                leftover = audio_chunk[n:]

            # Flush remaining samples with zero-padding
            if len(leftover) > 0:
                chunk = np.pad(leftover, (0, self.blocksize - len(leftover)))
                yield chunk

        except Exception as e:
            logger.error(f"MiniMax TTS error: {e}", exc_info=True)
        finally:
            self.should_listen.set()
