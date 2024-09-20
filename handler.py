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
        ) = get_default_arguments(mode='none', log_level='DEBUG')
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
        self.sessions = {}  # Store session information

    def _collect_output(self, session_id):
        while True:
            try:
                output = self.queues_and_events['send_audio_chunks_queue'].get(timeout=2)
                if isinstance(output, (str, bytes)) and output in (b"END", "END"):
                    self.sessions[session_id]['status'] = 'completed'
                    break
                elif isinstance(output, np.ndarray):
                    self.sessions[session_id]['chunks'].append(output.tobytes())
                else:
                    self.sessions[session_id]['chunks'].append(output)
            except Empty:
                continue

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        request_type = data.get("request_type", "start")
        
        if request_type == "start":
            return self._handle_start_request(data)
        elif request_type == "continue":
            return self._handle_continue_request(data)
        else:
            raise ValueError(f"Unsupported request type: {request_type}")

    def _handle_start_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'status': 'processing',
            'chunks': [],
            'last_sent_index': 0
        }

        input_type = data.get("input_type", "text")
        input_data = data.get("inputs", "")

        if input_type == "speech":
            audio_array = np.frombuffer(input_data, dtype=np.int16)
            self.queues_and_events['recv_audio_chunks_queue'].put(audio_array.tobytes())
        elif input_type == "text":
            self.queues_and_events['text_prompt_queue'].put(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # Start output collection in a separate thread
        threading.Thread(target=self._collect_output, args=(session_id,)).start()

        return {"session_id": session_id, "status": "processing"}

    def _handle_continue_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = data.get("session_id")
        if not session_id or session_id not in self.sessions:
            raise ValueError("Invalid or missing session_id")

        session = self.sessions[session_id]
        chunks_to_send = session['chunks'][session['last_sent_index']:]
        session['last_sent_index'] = len(session['chunks'])

        if chunks_to_send:
            combined_audio = b''.join(chunks_to_send)
            base64_audio = base64.b64encode(combined_audio).decode('utf-8')
            return {
                "session_id": session_id,
                "status": session['status'],
                "output": base64_audio
            }
        else:
            return {
                "session_id": session_id,
                "status": session['status'],
                "output": None
            }

    def cleanup(self):
        # Stop the pipeline
        self.pipeline_manager.stop()
        
        # Stop the output collector thread
        self.queues_and_events['send_audio_chunks_queue'].put(b"END")
        self.output_collector_thread.join()