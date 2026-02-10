from time import perf_counter
from baseHandler import BaseHandler
import numpy as np
import logging
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class PocketTTSHandler(BaseHandler):
    """
    Handler for Pocket TTS model from Kyutai Labs.
    Supports streaming audio generation with voice cloning.
    """

    def setup(
        self,
        should_listen,
        device="cpu",
        voice="alba",  # Default voice from catalog
        sample_rate=16000,  # Match the pipeline's audio output (LocalAudioStreamer uses 16kHz)
        blocksize=512,
        max_tokens=50,
        gen_kwargs=None,  # For compatibility with pipeline, not used
    ):
        """
        Initialize Pocket TTS handler.

        Args:
            should_listen: Event to control when to start listening again
            device: Device to run model on ('cpu', 'cuda', 'mps')
            voice: Voice to use. Can be:
                - A preset name: 'alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'
                - A local audio file path
                - A Hugging Face path like "hf://kyutai/tts-voices/..."
            sample_rate: Output sample rate (pocket-tts generates at 24kHz and will be resampled to this rate). Default 16kHz matches the pipeline's audio output.
            blocksize: Size of audio blocks to yield
            max_tokens: Maximum tokens to generate
        """
        self.should_listen = should_listen
        self.device = device
        self.voice = voice
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.max_tokens = max_tokens

        # Suppress verbose logging from pocket_tts library
        logging.getLogger("pocket_tts").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)

        # Import and load model
        from pocket_tts import TTSModel

        logger.info(f"Loading Pocket TTS model")
        self.model = TTSModel.load_model()

        # Move model to specified device
        if device == "cuda":
            self.model = self.model.cuda()
        elif device == "mps":
            self.model = self.model.to("mps")
        elif device != "cpu":
            self.model = self.model.to(device)

        logger.info(f"Pocket TTS model moved to {device}")

        # Load voice state
        logger.info(f"Loading voice: {voice}")
        self.voice_state = self.model.get_state_for_audio_prompt(voice)

        logger.info(f"Pocket TTS model sample rate: {self.model.sample_rate}")

    @property
    def min_time_to_debug(self):
        """
        Override to suppress logging for individual audio chunks.
        Pocket TTS yields many small chunks (~10-20ms each), which would flood logs.
        Only log if a chunk takes unusually long (>100ms).
        """
        return 0.1  # 100ms threshold

    def process(self, llm_sentence):
        """
        Process text from LLM and generate audio.

        Args:
            llm_sentence: Text to convert to speech, or tuple of (text, language_code)
        """
        # Handle tuple input (text, language_code)
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence
            logger.debug(f"Received language code: {language_code}")

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Generate audio stream
        logger.debug(f"Generating audio for: {llm_sentence[:50]}...")

        global pipeline_start
        first_chunk = True

        # Calculate target chunk size in original sample rate
        # We need enough samples so that after resampling we get blocksize samples
        needs_resampling = self.model.sample_rate != self.sample_rate
        if needs_resampling:
            from scipy.signal import resample_poly
            # Target chunk size in original sample rate (before resampling)
            resample_ratio = self.model.sample_rate / self.sample_rate
            target_chunk_size = int(self.blocksize * resample_ratio)
            # Calculate up/down factors for polyphase resampling
            # For 24kHz -> 16kHz: 16000/24000 = 2/3
            from math import gcd
            g = gcd(self.sample_rate, self.model.sample_rate)
            self._resample_up = self.sample_rate // g
            self._resample_down = self.model.sample_rate // g
        else:
            target_chunk_size = self.blocksize

        # Buffer to accumulate audio until we have enough for one block
        audio_buffer = []
        buffer_size = 0

        for audio_chunk in self.model.generate_audio_stream(
            self.voice_state,
            llm_sentence,
            max_tokens=self.max_tokens,
            copy_state=True,  # Don't modify the original voice state
        ):
            if first_chunk and "pipeline_start" in globals():
                logger.debug(
                    f"Time to first audio: {perf_counter() - pipeline_start:.3f}s"
                )
                first_chunk = False

            # Convert from torch tensor to numpy and add to buffer
            audio_np = audio_chunk.cpu().numpy()
            audio_buffer.append(audio_np)
            buffer_size += len(audio_np)

            # Process buffer when we have enough samples for at least one block
            while buffer_size >= target_chunk_size:
                # Concatenate buffer
                concatenated = np.concatenate(audio_buffer)

                # Extract enough samples for one block
                chunk_to_process = concatenated[:target_chunk_size]
                remainder = concatenated[target_chunk_size:]

                # Resample this chunk if needed
                if needs_resampling:
                    chunk_resampled = resample_poly(
                        chunk_to_process,
                        up=self._resample_up,
                        down=self._resample_down,
                    )
                else:
                    chunk_resampled = chunk_to_process

                # Convert to int16 format expected by audio output
                audio_int16 = (chunk_resampled * 32768).astype(np.int16)

                # Ensure exact blocksize (pad or trim if resampling caused slight size differences)
                if len(audio_int16) < self.blocksize:
                    audio_int16 = np.pad(audio_int16, (0, self.blocksize - len(audio_int16)))
                elif len(audio_int16) > self.blocksize:
                    audio_int16 = audio_int16[:self.blocksize]

                yield audio_int16

                # Update buffer with remainder
                audio_buffer = [remainder] if len(remainder) > 0 else []
                buffer_size = len(remainder)

        # Process any remaining audio in buffer
        if audio_buffer and buffer_size > 0:
            concatenated = np.concatenate(audio_buffer)

            # Resample the remainder
            if needs_resampling:
                concatenated = resample_poly(
                    concatenated,
                    up=self._resample_up,
                    down=self._resample_down,
                )

            # Convert to int16
            audio_int16 = (concatenated * 32768).astype(np.int16)

            # Yield in blocks
            for i in range(0, len(audio_int16), self.blocksize):
                chunk = audio_int16[i : i + self.blocksize]
                # Pad last chunk if needed
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        self.should_listen.set()
