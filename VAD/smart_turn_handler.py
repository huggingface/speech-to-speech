"""
Smart Turn v3 integration for speech-to-speech pipeline.

Uses pipecat-ai's Smart Turn v3 model (8MB ONNX, Whisper Tiny encoder + classifier)
to determine if a user has finished their turn based on audio prosody, not just silence.

Designed to work alongside Silero VAD:
- Silero VAD detects speech/silence boundaries
- Smart Turn analyzes the audio when silence is detected to decide if the turn is complete

Usage in pipeline:
    The SmartTurnHandler wraps the existing VADHandler. When Silero detects end-of-speech,
    Smart Turn runs on the accumulated audio. If the turn is incomplete, the pipeline
    continues listening instead of forwarding to STT.

Model: https://huggingface.co/pipecat-ai/smart-turn-v3
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to speech-to-speech root
DEFAULT_CPU_MODEL = "models/smart-turn-v3/smart-turn-v3.2-cpu.onnx"
DEFAULT_GPU_MODEL = "models/smart-turn-v3/smart-turn-v3.2-gpu.onnx"

MAX_AUDIO_SECONDS = 8
SAMPLE_RATE = 16000


class SmartTurnAnalyzer:
    """
    Analyzes audio to determine if a speaker has completed their turn.

    Uses Smart Turn v3 ONNX model (8MB int8 quantized for CPU, 32MB fp32 for GPU).
    Runs inference on up to 8 seconds of audio from the current turn.

    The model outputs a probability (0-1) where:
    - > threshold → turn complete (user finished speaking)
    - ≤ threshold → turn incomplete (user paused mid-thought)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5,
        base_dir: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to ONNX or TensorRT (.trt) model. Auto-selects if None.
            device: "cpu" or "cuda". CUDA uses TensorRT if .trt model available.
            threshold: Probability threshold for turn completion.
            base_dir: Base directory for resolving relative model paths.
        """
        from transformers import WhisperFeatureExtractor

        self.threshold = threshold
        self.device = device
        self.backend = None

        # Resolve model path
        if model_path is None:
            if device == "cuda":
                # Prefer TensorRT engine if available
                trt_path = DEFAULT_GPU_MODEL.replace(".onnx", ".trt")
                if base_dir:
                    trt_path = str(Path(base_dir) / trt_path)
                if Path(trt_path).exists():
                    model_path = trt_path
                else:
                    default = DEFAULT_GPU_MODEL
                    model_path = str(Path(base_dir) / default) if base_dir else default
            else:
                default = DEFAULT_CPU_MODEL
                model_path = str(Path(base_dir) / default) if base_dir else default

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Smart Turn model not found at {model_path}. "
                "Download from: https://huggingface.co/pipecat-ai/smart-turn-v3"
            )

        # Load model based on file type and device
        if model_path.endswith(".trt"):
            self._load_tensorrt(model_path)
        else:
            self._load_onnx(model_path, device)

        # Whisper feature extractor (chunk_length=8 for 8s max)
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)

        logger.info(
            f"Smart Turn v3 loaded: {model_path} ({self.backend}, threshold={threshold})"
        )

        # Warmup
        self._warmup()

    def _load_tensorrt(self, model_path: str):
        """Load TensorRT engine for GPU inference (~1.5ms)."""
        import tensorrt as trt
        import torch

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())

        self._trt_context = engine.create_execution_context()
        self._trt_context.set_input_shape("input_features", (1, 80, 800))

        self._trt_input = torch.empty((1, 80, 800), dtype=torch.float32, device="cuda")
        self._trt_output = torch.empty((1, 1), dtype=torch.float32, device="cuda")
        self._trt_stream = torch.cuda.Stream()

        self._trt_context.set_tensor_address("input_features", self._trt_input.data_ptr())
        self._trt_context.set_tensor_address("logits", self._trt_output.data_ptr())

        self.backend = "tensorrt"

    def _load_onnx(self, model_path: str, device: str):
        """Load ONNX model for CPU inference (~80ms)."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=providers
        )
        self.backend = "onnx"

    def _warmup(self):
        """Run a dummy inference to warm up the ONNX session."""
        dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.predict(dummy)
        logger.info("Smart Turn warmed up")

    def predict(self, audio_array: np.ndarray) -> dict:
        """
        Predict whether the user's turn is complete.

        Args:
            audio_array: PCM audio at 16kHz, float32. Can be any length;
                        will be truncated to last 8s or zero-padded.

        Returns:
            dict with:
                - complete: bool, True if turn is complete
                - probability: float, completion probability (0-1)
                - inference_ms: float, inference time in milliseconds
        """
        t0 = time.perf_counter()

        # Truncate to last 8 seconds or pad
        max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
        if len(audio_array) > max_samples:
            audio_array = audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            padding = max_samples - len(audio_array)
            audio_array = np.pad(
                audio_array, (padding, 0), mode="constant", constant_values=0
            )

        # Extract Whisper features
        return_tensors = "pt" if self.backend == "tensorrt" else "np"
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors=return_tensors,
            padding="max_length",
            max_length=MAX_AUDIO_SECONDS * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )

        if self.backend == "tensorrt":
            import torch

            self._trt_input.copy_(inputs.input_features)
            with torch.cuda.stream(self._trt_stream):
                self._trt_context.execute_async_v3(
                    stream_handle=self._trt_stream.cuda_stream
                )
            self._trt_stream.synchronize()
            probability = self._trt_output.cpu().item()
        else:
            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)
            outputs = self.session.run(None, {"input_features": input_features})
            probability = outputs[0][0].item()

        inference_ms = (time.perf_counter() - t0) * 1000

        return {
            "complete": probability > self.threshold,
            "probability": round(probability, 4),
            "inference_ms": round(inference_ms, 1),
        }


class SmartTurnVADHandler:
    """
    Drop-in wrapper for the speech-to-speech VADHandler that adds Smart Turn.

    When the underlying VAD detects end-of-speech:
    1. Run Smart Turn on the accumulated audio
    2. If turn complete → forward audio to STT (normal flow)
    3. If turn incomplete → continue listening (extend VAD timeout)

    This reduces false turn-taking on pauses, "um"s, and mid-thought breaks.
    """

    def __init__(
        self,
        vad_handler,
        smart_turn: SmartTurnAnalyzer,
        max_incomplete_extensions: int = 3,
        extension_timeout_s: float = 2.0,
    ):
        """
        Args:
            vad_handler: The existing VADHandler instance.
            smart_turn: SmartTurnAnalyzer instance.
            max_incomplete_extensions: Max times to extend listening on incomplete turns.
            extension_timeout_s: Additional silence to wait after incomplete detection.
        """
        self.vad = vad_handler
        self.smart_turn = smart_turn
        self.max_incomplete_extensions = max_incomplete_extensions
        self.extension_timeout_s = extension_timeout_s
        self._extension_count = 0

    def should_forward_to_stt(self, audio_array: np.ndarray) -> bool:
        """
        Called when VAD detects end-of-speech. Returns True if the turn
        should be forwarded to STT, False to continue listening.

        Args:
            audio_array: Full audio of the current turn (float32, 16kHz).

        Returns:
            True if turn is complete, False to keep listening.
        """
        if self._extension_count >= self.max_incomplete_extensions:
            logger.info(
                f"Smart Turn: max extensions ({self.max_incomplete_extensions}) reached, "
                "forwarding anyway"
            )
            self._extension_count = 0
            return True

        result = self.smart_turn.predict(audio_array)

        if result["complete"]:
            logger.info(
                f"Smart Turn: COMPLETE (p={result['probability']:.3f}, "
                f"{result['inference_ms']:.0f}ms)"
            )
            self._extension_count = 0
            return True
        else:
            self._extension_count += 1
            logger.info(
                f"Smart Turn: INCOMPLETE (p={result['probability']:.3f}, "
                f"{result['inference_ms']:.0f}ms, "
                f"extension {self._extension_count}/{self.max_incomplete_extensions})"
            )
            return False

    def reset(self):
        """Reset extension counter for a new turn."""
        self._extension_count = 0
