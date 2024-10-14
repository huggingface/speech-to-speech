import json
import logging
import time
from typing import Any, Dict, Generator, List

import numpy as np
from rich.console import Console
from transformers import pipeline

from baseHandler import BaseHandler

logger = logging.getLogger(__name__)
console = Console()


class Wav2Vec2STVHandler(BaseHandler):
    """
    Handles the Speech-To-Viseme generation using a Wav2Vec2 model for automatic
    speech recognition (ASR) and phoneme mapping to visemes.

    Attributes:
        MIN_AUDIO_LENGTH (float): Minimum length of audio (in seconds) required
                                  for phoneme extraction.
    """

    MIN_AUDIO_LENGTH = 0.5  # Minimum audio length in seconds for phoneme extraction

    def setup(
        self,
        should_listen: bool,
        model_name: str = "bookbot/wav2vec2-ljspeech-gruut",
        blocksize: int = 512,
        device: str = "cuda",
        skip: bool = False,
        gen_kwargs: Dict[str, Any] = {},  # Not used
    ) -> None:
        """
        Initializes the handler by loading the ASR model and phoneme-to-viseme map.

        Args:
            should_listen (bool): Flag indicating whether the speech-to-speech pipeline should start
                listening to the user or not.
            model_name (str): Name of the ASR model to use.
                Defaults to "bookbot/wav2vec2-ljspeech-gruut".
            blocksize (int): Size of each audio block when processing audio.
                Defaults to 512.
            device (str): Device to run the model on ("cuda", "mps", or "cpu").
                Defaults to "cuda".
            skip (bool): If True, the speech-to-viseme process is skipped.
                Defaults to False.
            gen_kwargs (dict): Additional parameters for speech generation.

        Returns:
            None
        """
        self.device = device
        self.gen_kwargs = gen_kwargs
        self.blocksize = blocksize
        self.should_listen = should_listen
        self.skip = skip

        # Load phoneme-to-viseme map from the JSON file
        # inspired by https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-ssml-phonetic-sets
        phoneme_viseme_map_file = "STV/phoneme_viseme_map.json"
        with open(phoneme_viseme_map_file, "r") as f:
            self.phoneme_viseme_map = json.load(f)

        # Initialize the ASR pipeline using the specified model and device
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype="auto",
        )
        self.expected_sampling_rate = self.asr_pipeline.feature_extractor.sampling_rate

        # Initialize an empty dictionary to store audio batch data
        self.audio_batch = {
            "waveform": np.array([]),
            "sampling_rate": self.expected_sampling_rate,
        }
        self.text_batch = None
        self.should_listen_flag = False

        self.warmup()  # Perform model warmup

    def warmup(self) -> None:
        """Warms up the model with dummy input to prepare it for inference.

        Returns:
            None
        """
        logger.info(f"Warming up {self.__class__.__name__}")
        start_time = time.time()

        # Create dummy input for warmup inference
        dummy_input = np.random.randn(self.blocksize).astype(np.int16)
        _ = self.speech_to_visemes(dummy_input)

        warmup_time = time.time() - start_time
        logger.info(
            f"{self.__class__.__name__}: warmed up in {warmup_time:.4f} seconds!"
        )

    def speech_to_visemes(self, audio: Any) -> List[Dict[str, Any]]:
        """
        Converts speech audio to visemes by performing Automatic Speech Recognition (ASR)
        and mapping phonemes to visemes.

        Args:
            audio (Any): The input audio data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing mapped visemes
                                  and their corresponding timestamps.

        Note:
            Heuristically, the input audio should be at least 0.5 seconds long for proper phoneme extraction.
        """

        def _map_phonemes_to_visemes(
            data: Dict[str, Any],
        ) -> List[Dict[str, Any]]:
            """
            Maps extracted phonemes to their corresponding visemes based on a predefined map.

            Args:
                data (Dict[str, Any]): Dictionary containing phoneme data where data['chunks']
                                    holds a list of phonemes and their timestamps.

            Returns:
                List[Dict[str, Any]]: A list of dictionaries with viseme IDs and their corresponding timestamps.
            """
            viseme_list = []
            chunks = data.get("chunks", [])

            # Map each phoneme to corresponding visemes
            for chunk in chunks:
                phoneme = chunk.get("text", None)
                timestamp = chunk.get("timestamp", None)
                visemes = self.phoneme_viseme_map.get(phoneme, [])

                for viseme in visemes:
                    viseme_list.append({"viseme": viseme, "timestamp": timestamp})

            return viseme_list

        # Perform ASR to extract phoneme data, including timestamps
        try:
            asr_result = self.asr_pipeline(audio, return_timestamps="char")
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return []
        # Map the phonemes obtained from ASR to visemes
        return _map_phonemes_to_visemes(asr_result)

    def process(self, data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Processes an audio file to generate visemes and output blocks of audio data
        along with corresponding viseme data.

        Args:
            data (Dict[str, Any]): Dictionary containing audio, text, and potentially additional information.

        Yields:
            Dict: A dictionary containing audio waveform, and optionally viseme data, text, and potentially additional information.
        """

        if "sentence_end" in data and data["sentence_end"]:
            self.should_listen_flag = True
        if self.skip:  # Skip viseme extraction if the flag is set
            yield {
                "audio": {
                    "waveform": data["audio"]["waveform"],
                    "sampling_rate": data["audio"]["sampling_rate"],
                },
                "text": data["text"] if "text" in data else None,
            }
        else:
            # Check if text data is present and save it for later
            if "text" in data and data["text"] is not None:
                self.text_batch = data["text"]
            # Concatenate new audio data into the buffer if available and valid
            if "audio" in data and data["audio"] is not None:
                audio_data = data["audio"]
                # Check if the sampling rate is valid and matches the expected one
                if audio_data.get("sampling_rate", None) != self.expected_sampling_rate:
                    logger.error(
                        f"Expected sampling rate {self.expected_sampling_rate}, "
                        f"but got {audio_data['sampling_rate']}."
                    )
                    return
                # Append the waveform to the audio buffer
                self.audio_batch["waveform"] = np.concatenate(
                    (self.audio_batch["waveform"], audio_data["waveform"]), axis=0
                )

            # Ensure the total audio length is sufficient for phoneme extraction
            if (
                len(self.audio_batch["waveform"]) / self.audio_batch["sampling_rate"]
                < self.MIN_AUDIO_LENGTH
            ):
                return
            else:
                logger.debug("Starting viseme inference...")

                # Perform viseme inference using the accumulated audio batch
                viseme_data = self.speech_to_visemes(self.audio_batch["waveform"])
                logger.debug("Viseme inference completed.")

                # Print the visemes and timestamps to the console
                for viseme in viseme_data:
                    console.print(
                        f"[blue]ASSISTANT_MOUTH_SHAPE: {viseme['viseme']} -- {viseme['timestamp']}"
                    )

                # Process the audio in chunks of the defined blocksize
                self.audio_batch["waveform"] = self.audio_batch["waveform"].astype(
                    np.int16
                )
                for i in range(0, len(self.audio_batch["waveform"]), self.blocksize):
                    chunk_waveform = self.audio_batch["waveform"][
                        i : i + self.blocksize
                    ]
                    padded_waveform = np.pad(
                        chunk_waveform, (0, self.blocksize - len(chunk_waveform))
                    )

                    chunk_data = {
                        "audio": {
                            "waveform": padded_waveform,
                            "sample_rate": self.audio_batch["sampling_rate"],
                        }
                    }

                    # Add text and viseme data only in the first chunk
                    if i == 0:
                        if self.text_batch:
                            chunk_data["text"] = self.text_batch
                        if viseme_data and len(viseme_data) > 0:
                            chunk_data["visemes"] = viseme_data
                    yield chunk_data

                # Reset the audio and text buffer after processing
                self.audio_batch = {
                    "waveform": np.array([]),
                    "sampling_rate": self.expected_sampling_rate,
                }
                self.text_batch = ""
        
        if self.should_listen_flag:
            self.should_listen.set()
            self.should_listen_flag = False
