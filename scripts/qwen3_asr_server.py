"""Minimal HTTP server wrapping Qwen3ASRModel.from_pretrained (transformers backend).

Exposes the same /v1/chat/completions contract that
speech_to_speech.STT.qwen3_asr_http_handler expects. Use this instead of `qwen-asr-serve`,
which wraps `vllm serve` and requires the vllm package -- unavailable on macOS/Apple Silicon
(no wheels), and generally unnecessary for a single-utterance-at-a-time voice pipeline.

Run this in its own virtualenv pinned to transformers==4.57.6 (the version `qwen-asr`
actually needs), separate from the main speech-to-speech install. See
src/speech_to_speech/STT/README.md for the full writeup.

Usage:
    pip install qwen-asr==0.0.6
    python scripts/qwen3_asr_server.py --device mps --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import time

import torch
from flask import Flask, jsonify, request
from qwen_asr import Qwen3ASRModel

app = Flask(__name__)
model: Qwen3ASRModel | None = None


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    payload = request.get_json(force=True)
    content = payload["messages"][0]["content"]
    audio_item = next(item for item in content if item.get("type") == "audio_url")
    audio_url = audio_item["audio_url"]["url"]

    # qwen_asr.inference.utils.load_audio_any accepts this string directly (http(s) URL,
    # local path, or a "data:audio/..." base64 URI) -- no manual decoding needed here.
    start = time.perf_counter()
    results = model.transcribe(audio=audio_url, language=None)
    elapsed_s = time.perf_counter() - start
    result = results[0]
    content_text = f"language {result.language}<asr_text>{result.text}" if result.language else result.text

    print(f"[qwen3_asr_server] transcribe took {elapsed_s:.3f}s -> {content_text!r}")
    return jsonify({"choices": [{"message": {"content": content_text}}]})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def main() -> None:
    global model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B", help="HF model id or local path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", help="cuda, mps, or cpu")
    parser.add_argument("--dtype", default="bfloat16", help="torch dtype, e.g. bfloat16, float16, float32")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device} ({args.dtype})...")
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=getattr(torch, args.dtype),
        device_map=args.device,
    )
    print(f"Ready. Listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
