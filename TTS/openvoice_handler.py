import torch
import os
import tempfile
import numpy as np
import librosa
from rich.console import Console
import logging

from baseHandler import BaseHandler

# Conditional imports to handle potential missing dependencies
try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter
    from melo.api import TTS
    OPENVOICE_AVAILABLE = True
except ImportError as e:
    OPENVOICE_AVAILABLE = False
    logging.warning(f"OpenVoice or MeloTTS not installed, OpenVoice handler will not be available: {e}")

logger = logging.getLogger(__name__)
console = Console()

# In the container, this will be /root/.cache/openvoice
CACHE_BASE_PATH = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
OPENVOICE_CACHE_DIR = os.path.join(CACHE_BASE_PATH, "openvoice")

class OpenVoiceHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        reference_audio=os.path.join(OPENVOICE_CACHE_DIR, "assets/resources/example_reference.mp3"),
        base_speaker="EN-US", # Use MeloTTS speaker IDs e.g., EN-US, EN-BR, ZH, ES, FR
        language="English", # This is for MeloTTS, not used by OpenVoice directly
        speed=1.0,
        chunk_size=512,
        gen_kwargs={},  # Unused
    ):
        if not OPENVOICE_AVAILABLE:
            raise ImportError("OpenVoice or MeloTTS not installed. Please ensure it is included in your requirements.")

        self.should_listen = should_listen
        self.device = device
        self.speed = speed
        self.chunk_size = chunk_size
        self.base_speaker_name = base_speaker
        self.language = language

        checkpoint_dir = os.path.join(OPENVOICE_CACHE_DIR, "checkpoints")

        # Determine which base speaker model to load (EN or ZH)
        if 'zh' in self.base_speaker_name.lower():
            lang_code = 'ZH'
        else:
            lang_code = 'EN'

        # 1. Initialize MeloTTS for base audio generation
        logger.info(f"Loading MeloTTS base speaker for language: {lang_code}")
        self.melo_model = TTS(language=lang_code, device=self.device)
        # Get the internal speaker ID for Melo
        self.melo_speaker_id = self.melo_model.hps.data.spk2id[self.base_speaker_name]

        # 2. Initialize OpenVoice ToneColorConverter
        converter_config_path = f'{checkpoint_dir}/converter/config.json'
        converter_ckpt_path = f'{checkpoint_dir}/converter/checkpoint.pth'
        logger.info(f"Loading ToneColorConverter from: {converter_ckpt_path}")
        self.tone_color_converter = ToneColorConverter(converter_config_path, device=self.device)
        self.tone_color_converter.load_ckpt(converter_ckpt_path)

        # 3. Load Source and Target Speaker Embeddings
        logger.info("Loading SE extractor")
        self.se_extractor = se_extractor.get_se_extractor(0)

        # The speaker key for the .pth file should be lowercase, e.g., 'en-us'
        source_speaker_key = self.base_speaker_name.lower().replace('_', '-')
        source_se_path = f'{checkpoint_dir}/base_speakers/ses/{source_speaker_key}.pth'
        logger.info(f"Loading base speaker embedding from: {source_se_path}")
        self.source_se = torch.load(source_se_path, map_location=self.device)

        logger.info(f"Processing reference audio for voice cloning: {reference_audio}")
        self.target_se, _ = self.se_extractor.get_se(
            reference_audio,
            self.tone_color_converter,
            target_dir=tempfile.gettempdir(),
            vad=True
        )
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        self.process("This is a warmup sentence.")
        logger.info(f"{self.__class__.__name__} warmed up!")

    def process(self, llm_sentence):
        if isinstance(llm_sentence, tuple):
            llm_sentence, _ = llm_sentence

        console.print(f"[green]ASSISTANT (OpenVoice): {llm_sentence}")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_audio_path = os.path.join(tmpdir, "base.wav")
            final_audio_path = os.path.join(tmpdir, "final.wav")

            # Stage 1: Generate base audio with MeloTTS
            self.melo_model.tts_to_file(
                llm_sentence,
                self.melo_speaker_id,
                base_audio_path,
                speed=self.speed
            )

            # Stage 2: Convert the base audio's voice to the reference voice
            self.tone_color_converter.convert(
                audio_src_path=base_audio_path,
                src_se=self.source_se,
                tgt_se=self.target_se,
                output_path=final_audio_path,
                message="@MyShell"  # Watermark as in the official example
            )

            # Read the final audio and stream it back
            audio_data, _ = librosa.load(final_audio_path, sr=16000)
            audio_data = (audio_data * 32768).astype(np.int16)

            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                yield chunk

        self.should_listen.set()
