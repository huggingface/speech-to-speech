from typing import Dict, Any, List, Generator
import torch
import os
import logging
from s2s_pipeline import main, prepare_all_args, get_default_arguments, setup_logger, initialize_queues_and_events, build_pipeline
import numpy as np
from queue import Queue, Empty
import threading

class EndpointHandler:
    def __init__(self, path=""):
        (
            self.module_kwargs,
            self.socket_receiver_kwargs,
            self.socket_sender_kwargs,
            self.vad_handler_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
        ) = get_default_arguments(device='cpu', mode='none', tts='melo')
        setup_logger(self.module_kwargs.log_level)

        prepare_all_args(
            self.module_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
        )

        self.queues_and_events = initialize_queues_and_events()

        self.pipeline_manager = build_pipeline(
            self.module_kwargs,
            self.socket_receiver_kwargs,
            self.socket_sender_kwargs,
            self.vad_handler_kwargs,
            self.whisper_stt_handler_kwargs,
            self.paraformer_stt_handler_kwargs,
            self.language_model_handler_kwargs,
            self.mlx_language_model_handler_kwargs,
            self.parler_tts_handler_kwargs,
            self.melo_tts_handler_kwargs,
            self.chat_tts_handler_kwargs,
            self.queues_and_events,
        )

        self.pipeline_manager.start()

        # Add a new queue for collecting the final output
        self.final_output_queue = Queue()

    def _collect_output(self):
        while True:
            try:
                output = self.queues_and_events['send_audio_chunks_queue'].get(timeout=5)  # 2-second timeout
                if isinstance(output, (str, bytes)) and output in (b"END", "END"):
                    self.final_output_queue.put("END")
                    break
                elif isinstance(output, np.ndarray):
                    self.final_output_queue.put(output.tobytes())
                else:
                    self.final_output_queue.put(output)
            except Empty:
                # If no output for 2 seconds, assume processing is complete
                self.final_output_queue.put("END")
                break

    def __call__(self, data: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Args:
            data (Dict[str, Any]): The input data containing the necessary arguments.
        
        Returns:
            Generator[Dict[str, Any], None, None]: A generator yielding output chunks from the model or pipeline.
        """
        # Start a thread to collect the final output
        self.output_collector_thread = threading.Thread(target=self._collect_output)
        self.output_collector_thread.start()

        input_type = data.get("input_type", "text")
        input_data = data.get("input", "")

        if input_type == "speech":
            # Convert input audio data to numpy array
            audio_array = np.frombuffer(input_data, dtype=np.int16)
            
            # Put audio data into the recv_audio_chunks_queue
            self.queues_and_events['recv_audio_chunks_queue'].put(audio_array.tobytes())
        elif input_type == "text":
            # Put text data directly into the text_prompt_queue
            self.queues_and_events['text_prompt_queue'].put(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # Collect all output chunks
        output_chunks = []
        while True:
            chunk = self.final_output_queue.get()
            if chunk == "END":
                break
            output_chunks.append(chunk)

        # Combine all audio chunks into a single byte string
        combined_audio = b''.join(output_chunks)

        return {"output": combined_audio}

    def cleanup(self):
        # Stop the pipeline
        self.pipeline_manager.stop()
        
        # Stop the output collector thread
        self.queues_and_events['send_audio_chunks_queue'].put(b"END")
        self.output_collector_thread.join()