import io
import time
import logging
import soundfile as sf
from baseHandler import BaseHandler
from typing import Generator, Optional, Tuple, Union
from const import OPENAI_API_KEY

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAITTSHandler(BaseHandler):
    """
    OpenAI Whisper / GPT-4o Transcription Handler (non-streaming chunks)
    """

    def setup(
        self,
        model: str = "gpt-4o-mini-transcribe",
        language: Optional[str] = None,
    ):
        self.model = model
        self.language = language
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY
        )

    # ── core ──────────────────────────────────────────────────────────────────
    def process(self, audio_chunk: Union[np.ndarray, bytes]) -> Generator[Tuple[str, str], None, None]:
        logger.info("OpenAI TTS Recieved Data!")
        try:
            # debug Set to English for now
            buffer = self._numpy_to_wav_buffer(audio_chunk)
            response = self.openai_client.audio.transcriptions.create(
                model=self.model,
                file=buffer,
                language="en",
            )
            yield response.text
        except Exception as e:
            logger.error("❌ OpenAI TTS API request failed: %s", e)
            yield ""

    # ── helpers ──────────────────────────────────────────────────────────────
    def _numpy_to_wav_buffer(self, audio_np: np.ndarray) -> io.BytesIO:
        """
        Converts a NumPy array (16 kHz mono) to an in-memory WAV file.
        Returns a BytesIO object *with* a `.name` so the OpenAI client
        can infer the MIME type.
        """
        buf = io.BytesIO()
        sf.write(buf, audio_np, 16000, format="WAV", subtype="PCM_16")
        with open(f"{int(time.time())}.wav", "wb") as f:
            sf.write(f, audio_np, 16000, format="WAV", subtype="PCM_16")
        buf.seek(0)
        buf.name = "chunk.wav"
        return buf