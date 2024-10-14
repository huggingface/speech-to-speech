from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import librosa
from rich.console import Console
from baseHandler import BaseHandler
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE = {
    "en": "eng", # English
    "fr": "fra", # French
    "es": "spa", # Spanish
    "ko": "kor", # Korean
    "hi": "hin", # Hindi
    "ar": "ara", # Arabic
    "ar": "hyw", # Armenian
    "az": "azb", # Azerbaijani
    "bu": "bul", # Bulgarian
    "ca": "cat", # Catalan
    "nl": "nld", # Dutch
    "fi": "fin", # Finnish
    "fr": "fra", # French
    "de": "deu", # German
    "el": "ell", # Greek
    "he": "heb", # Hebrew
    "hu": "hun", # Hungarian
    "is": "isl", # Icelandic
    "id": "ind", # Indonesian
    "ka": "kan", # Kannada
    "kk": "kaz", # Kazakh
    "lv": "lav", # Latvian
    "zl": "zlm", # Malay
    "ma": "mar", # Marathi
    "fa": "fas", # Persian
    "po": "pol", # Polish
    "pt": "por", # Portuguese
    "ro": "ron", # Romanian
    "ru": "rus", # Russian
    "sw": "swh", # Swahili
    "sv": "swe", # Swedish
    "tg": "tgl", # Tagalog
    "ta": "tam", # Tamil
    "th": "tha", # Thai
    "tu": "tur", # Turkish
    "uk": "ukr", # Ukrainian
    "ur": "urd", # Urdu
    "vi": "vie", # Vietnamese
    "cy": "cym", # Welsh
}

class FacebookMMSTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        torch_dtype="float32",
        language="en",
        stream=True,
        chunk_size=512,
        **kwargs
    ):
        self.should_listen = should_listen
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.stream = stream
        self.chunk_size = chunk_size
        self.language = language

        self.load_model(self.language)
        self.warmup()

    def load_model(self, language_code):
        try:
            model_name = f"facebook/mms-tts-{WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE[language_code]}"
            logger.info(f"Loading model: {model_name}")
            self.model = VitsModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language = language_code
        except KeyError:
            logger.warning(f"Unsupported language: {language_code}. Falling back to English.")
            self.load_model("en")

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        output = self.generate_audio("Hello, this is a test")

    def generate_audio(self, text):
        if not text:
            logger.warning("Received empty text input")
            return None

        try:
            logger.debug(f"Tokenizing text: {text}")
            logger.debug(f"Current language: {self.language}")
            logger.debug(f"Tokenizer: {self.tokenizer}")
            
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device).long()
            attention_mask = inputs.attention_mask.to(self.device)
            
            logger.debug(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
            logger.debug(f"Input IDs: {input_ids}")
            
            if input_ids.numel() == 0:
                logger.error("Input IDs tensor is empty")
                return None

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            logger.debug(f"Output waveform shape: {output.waveform.shape}")
            return output.waveform
        except Exception as e:
            logger.error(f"Error in generate_audio: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def process(self, llm_sentence):
        language_code = None

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")
        logger.debug(f"Processing text: {llm_sentence}")
        logger.debug(f"Language code: {language_code}")

        if language_code is not None and self.language != language_code:
            try:
                logger.info(f"Switching language from {self.language} to {language_code}")
                self.load_model(language_code)
            except KeyError:
                console.print(f"[red]Language {language_code} not supported by Facebook MMS. Using {self.language} instead.")
                logger.warning(f"Unsupported language: {language_code}")

        audio_output = self.generate_audio(llm_sentence)
        
        if audio_output is None or audio_output.numel() == 0:
            logger.warning("No audio output generated")
            self.should_listen.set()
            return

        audio_numpy = audio_output.cpu().numpy().squeeze()
        logger.debug(f"Raw audio shape: {audio_numpy.shape}, dtype: {audio_numpy.dtype}")
        
        audio_resampled = librosa.resample(audio_numpy, orig_sr=self.model.config.sampling_rate, target_sr=16000)
        logger.debug(f"Resampled audio shape: {audio_resampled.shape}, dtype: {audio_resampled.dtype}")
        
        audio_int16 = (audio_resampled * 32768).astype(np.int16)
        logger.debug(f"Final audio shape: {audio_int16.shape}, dtype: {audio_int16.dtype}")

        if self.stream:
            for i in range(0, len(audio_int16), self.chunk_size):
                chunk = audio_int16[i:i + self.chunk_size]
                yield np.pad(chunk, (0, self.chunk_size - len(chunk)))
        else:
            for i in range(0, len(audio_int16), self.chunk_size):
                yield np.pad(
                    audio_int16[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_int16[i : i + self.chunk_size])),
                )

        self.should_listen.set()