from melo.api import TTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch
from langdetect import detect
from datetime import datetime

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN",
    #"fr": "FR",
    #"es": "ES",
    #"zh": "ZH",
    #"ja": "JP",
    #"ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-BR",
    #"fr": "FR",
    #"es": "ES",
    #"zh": "ZH",
    #"ja": "JP",
    #"ko": "KR",
}

class MeloTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="mps",
        language="en",
        speaker_to_id="en",
        gen_kwargs={},
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        self.blocksize = blocksize
        self.initialize_model(language, speaker_to_id)
        self.warmup()

    def initialize_model(self, language, speaker_to_id):
        logger.debug(f"Initializing model: {language}")
        self.model = TTS(
            language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[language], device=self.device
        )
        self.speaker_id = self.model.hps.data.spk2id[
            WHISPER_LANGUAGE_TO_MELO_SPEAKER[speaker_to_id]
        ]

    def warmup(self):
        logger.debug(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("Text", self.speaker_id, quiet=True)

    def detect_language(self, text):
        try:
            detected = detect(text)
            return detected if detected in WHISPER_LANGUAGE_TO_MELO_LANGUAGE else 'en'
        except:
            return 'en'

    def process(self, llm_sentence):
        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
        
        if language_code is None:
            language_code = self.detect_language(llm_sentence)

        tts_timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        console.print(f"[blue]POLY ({language_code.upper()})[{tts_timestamp}]: {llm_sentence}")

        if language_code != self.language:
            logger.info(f"Switching language: {self.language} -> {language_code}")
            try:
                self.initialize_model(language_code, language_code)
                self.language = language_code
            except KeyError:
                logger.warning(f"Unsupported language: {language_code}. Using {self.language}.")

        if self.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        try:
            audio_chunk = self.model.tts_to_file(
                llm_sentence, self.speaker_id, quiet=True
            )
        except (AssertionError, RuntimeError) as e:
            logger.error(f"TTS error: {e}")
            audio_chunk = np.array([])

        if len(audio_chunk) == 0:
            self.should_listen.set()
            return

        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)

        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()

def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("melo").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
