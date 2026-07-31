FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheel first: plain `pip install torch` on Linux defaults to the
# CUDA-bundled wheel (multi-GB) even though this image has no GPU to use it with.
RUN pip install --no-cache-dir --default-timeout=180 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

# Normal install (no --no-deps): this container is isolated from the main speech-to-speech
# install, so qwen-asr's own transformers==4.57.6 pin resolves cleanly here. See
# src/speech_to_speech/STT/README.md for why this can't share a process/venv with the
# transformers==5.6.2 the main pipeline pins on macOS.
RUN pip install --no-cache-dir --default-timeout=180 --retries 5 "qwen-asr==0.0.6" flask

# Not `qwen-asr-serve`: it wraps `vllm serve` and requires the vllm package, which has no
# wheels for macOS/Apple Silicon (and is overkill for one utterance at a time). This runs
# qwen3_asr_server.py instead -- a small Flask server around the same
# Qwen3ASRModel.from_pretrained(...).transcribe(...) call, exposing the same
# /v1/chat/completions contract this project's Qwen3ASRHTTPSTTHandler expects.
COPY scripts/qwen3_asr_server.py .

EXPOSE 8000

ENTRYPOINT ["python", "qwen3_asr_server.py"]
CMD ["--model", "Qwen/Qwen3-ASR-1.7B", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
