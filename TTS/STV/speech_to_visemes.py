"""This module contains the SpeechToVisemes class, which handles the conversion of speech to visemes."""
from typing import List, Dict, Any
from transformers import pipeline
import logging
import json

logger = logging.getLogger(__name__)

class SpeechToVisemes():
    """
    Handles the conversion of speech to visemes using a phoneme-to-viseme mapping.

    Attributes:
        model_name (str): The name of the model to use for speech recognition.
        device (str): The device to run the model on (e.g., "cpu", "mps", "cuda").
        gen_kwargs (dict): Additional generation parameters for the speech recognition pipeline.
        asr_pipeline (transformers.Pipeline): The automatic speech recognition pipeline.
    """

    def __init__(
        self,
        model_name="bookbot/wav2vec2-ljspeech-gruut",
        device="mps",
        gen_kwargs={},
    ):
        """
        Initializes the SpeechToVisemes class with the specified parameters.

        Args:
            model_name (str, optional): The name of the model to use for speech recognition.
            device (str, optional): The device to run the model on.
            gen_kwargs (dict, optional): Additional generation parameters for the speech recognition pipeline.
        """
        self.device = device
        self.gen_kwargs = gen_kwargs

        # This dictionary is inspired by https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-ssml-phonetic-sets
        phoneme_viseme_map_file="TTS/STV/phoneme_viseme_map.json"
        with open(phoneme_viseme_map_file, 'r') as f:
            self.phoneme_viseme_map = json.load(f)

        # Initialize the automatic speech recognition pipeline
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition", model=model_name, device=device
        )

    def _map_phonemes_to_visemes(
        self, 
        data: Dict[str, Any], 
    ) -> List[Dict[str, Any]]:
        """
        Maps phonemes to corresponding visemes with timestamps.

        Refer to the following references for more information on the phoneme-to-viseme mapping:
            - https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-speech-synthesis-viseme?tabs=visemeid&pivots=programming-language-python#map-phonemes-to-visemes
            - https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-ssml-phonetic-sets

        Args:
            data (Dict[str, Any]): A dictionary containing phoneme data, where data['chunks'] 
                holds a list of phonemes and their timestamps.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the viseme ID 
                and the corresponding timestamp.
        """
        viseme_list = []
        chunks = data.get('chunks', [])

        for _, chunk in enumerate(chunks):
            phoneme = chunk.get('text', None)
            timestamp = chunk.get('timestamp', None)
            visemes = self.phoneme_viseme_map.get(phoneme, [])
            
            for viseme in visemes:
                viseme_list.append({
                    'viseme': viseme,
                    'timestamp': timestamp
                })

        return viseme_list


    def process(self, audio_file: str) -> List[Dict[str, Any]]:
        """Process an audio file and convert speech to visemes.
        
        Heuristically, we found that the model requires at least 0.5 seconds of audio to run phoneme recognition.
        This value may be also depended on the model, the language, and other factors.

        Args:
            audio_file (str): The path to the audio file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the viseme 
                ID and the corresponding timestamp.
        """
        # Perform ASR to get phoneme data
        asr_result = self.asr_pipeline(audio_file, return_timestamps='char')
        # Map phonemes to visemes
        viseme_data = self._map_phonemes_to_visemes(asr_result)
        return viseme_data
    