from typing import Dict, Any, List, Generator
import torch
import os
import logging
from s2s_pipeline import main, prepare_all_args, get_default_arguments, setup_logger, initialize_queues_and_events, build_pipeline
import numpy as np
from queue import Queue, Empty
import threading
import base64
import uuid
import torch

class EndpointHandler:
    def __init__(self, path=""):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lm_model_name = os.getenv('LM_MODEL_NAME', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        chat_size = int(os.getenv('CHAT_SIZE', 10))

        (
            self.module_kwargs,
            self.socket_receiver_kwargs,
            self.socket_sender_kwargs,
            self.vad_handler_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.faster_whisper_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.open_api_language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
            self.facebook_mm_stts_handler_kwargs,
        ) = get_default_arguments(mode='none', log_level='DEBUG', lm_model_name=lm_model_name, 
            tts="melo", device=device, chat_size=chat_size)
        setup_logger(self.module_kwargs.log_level)

        prepare_all_args(
            self.module_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.faster_whisper_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.open_api_language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
            self.facebook_mm_stts_handler_kwargs,
        )

        self.queues_and_events = initialize_queues_and_events()

        self.pipeline_manager = build_pipeline(
            self.module_kwargs,
            self.socket_receiver_kwargs,
            self.socket_sender_kwargs,
            self.vad_handler_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.faster_whisper_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.open_api_language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
            self.facebook_mm_stts_handler_kwargs,
            self.queues_and_events,
        )

        self.vad_chunk_size = 512  # Set the chunk size required by the VAD model
        self.sample_rate = 16000  # Set the expected sample rate

    def process_streaming_data(self, data: bytes) -> bytes:
        audio_array = np.frombuffer(data, dtype=np.int16)

        # Process the audio data in chunks
        chunks = [audio_array[i:i+self.vad_chunk_size] for i in range(0, len(audio_array), self.vad_chunk_size)]
        
        for chunk in chunks:
            if len(chunk) == self.vad_chunk_size:
                self.queues_and_events['recv_audio_chunks_queue'].put(chunk.tobytes())
            elif len(chunk) < self.vad_chunk_size:
                # Pad the last chunk if it's smaller than the required size
                padded_chunk = np.pad(chunk, (0, self.vad_chunk_size - len(chunk)), 'constant')
                self.queues_and_events['recv_audio_chunks_queue'].put(padded_chunk.tobytes())

        # Collect the output, if any
        try:
            output = self.queues_and_events['send_audio_chunks_queue'].get_nowait()  # improvement idea, group all available output chunks
            if isinstance(output, np.ndarray):
                return output.tobytes()
            else:
                return output
        except Empty:
            return None

    def cleanup(self):
        # Stop the pipeline
        self.pipeline_manager.stop()
        
        # Stop the output collector thread
        self.queues_and_events['send_audio_chunks_queue'].put(b"END")
        self.output_collector_thread.join()