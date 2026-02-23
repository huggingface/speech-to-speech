#!/usr/bin/env python3
"""
True end-to-end speech-to-speech cascade benchmark.

Pipeline: Audio in → STT (Parakeet) → LLM (streaming) → TTS (CUDA graphs) → Audio out

All stages run in a single process. Measures wall-clock time from
"user stops speaking" to "first audio chunk ready".
"""

import argparse
import json
import os
import re
import time
from time import perf_counter

import numpy as np
import torch
import torch_distributed_stub  # noqa: F401 — stub torch.distributed for Jetson
import soundfile as sf

# ── Config ──────────────────────────────────────────────────────────────────

TTS_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TTS_REF_AUDIO = "ref_audio.wav"
TTS_REF_TEXT = "This is a reference audio sample for voice cloning."

LLM_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Respond concisely in 1-2 sentences. "
    "Your output will be spoken aloud, so be natural and conversational."
)

SENTENCE_ENDERS = re.compile(r'[.!?]')


# ── STT ─────────────────────────────────────────────────────────────────────

def load_stt_model():
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
    # Disable NeMo's internal CUDA graphs for decoding — avoids NVRTC/cuda-python
    # version conflicts. Our TTS CUDA graphs are separate and unaffected.
    try:
        comp = model.decoding.decoding._decoding_computer
        comp.cuda_graphs_mode = None  # Forces loop_labels_torch path in __call__
    except Exception:
        pass
    model.eval()
    model = model.cuda()
    return model


def run_stt(model, audio_path: str) -> tuple:
    """Run Parakeet STT. Returns (text, inference_time)."""
    t0 = perf_counter()
    transcriptions = model.transcribe([audio_path])
    inference_time = perf_counter() - t0

    if isinstance(transcriptions, list):
        text = transcriptions[0] if transcriptions else ""
    elif hasattr(transcriptions, 'text'):
        text = transcriptions.text[0] if hasattr(transcriptions.text, '__getitem__') else str(transcriptions.text)
    else:
        text = str(transcriptions)

    # Clean up common NeMo output formatting
    text = text.strip().strip("[]'\"")

    return text, inference_time


# ── LLM (streaming, yields sentences) ─────────────────────────────────────

_local_llm_model = None
_local_llm_tokenizer = None
_local_llm_pipe = None


def load_local_llm(model_name: str, load_in_4bit: bool = False):
    """Load a local transformers LLM. Cached after first call."""
    global _local_llm_model, _local_llm_tokenizer, _local_llm_pipe
    if _local_llm_pipe is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    print(f"Loading local LLM: {model_name} (4-bit={load_in_4bit})...")

    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["device_map"] = "cuda"

    _local_llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _local_llm_model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    _local_llm_pipe = pipeline(
        "text-generation", model=_local_llm_model, tokenizer=_local_llm_tokenizer,
    )
    print(f"Local LLM loaded: {model_name}")


def stream_local_llm_sentences(text: str, model: str = None, no_think: bool = False):
    """Stream sentences from a local transformers model. Yields (sentence, elapsed, ttft)."""
    from threading import Thread
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        _local_llm_tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    gen_kwargs = {
        "streamer": streamer,
        "max_new_tokens": 150,
        "return_full_text": False,
    }
    if no_think:
        gen_kwargs["tokenizer_encode_kwargs"] = {"enable_thinking": False}

    t0 = perf_counter()
    thread = Thread(target=_local_llm_pipe, args=(messages,), kwargs=gen_kwargs)
    thread.start()

    buffer = ""
    ttft = None

    for new_text in streamer:
        if not new_text:
            continue
        if ttft is None:
            ttft = perf_counter() - t0
        buffer += new_text

        match = SENTENCE_ENDERS.search(buffer)
        if match:
            idx = match.end()
            while True:
                m = SENTENCE_ENDERS.search(buffer, idx)
                if m:
                    idx = m.end()
                else:
                    break
            sentence = buffer[:idx].strip()
            buffer = buffer[idx:]
            if sentence:
                yield sentence, perf_counter() - t0, ttft

    thread.join()
    if buffer.strip():
        yield buffer.strip(), perf_counter() - t0, ttft


def stream_llm_sentences(text: str, model: str = "gpt-4o-mini", base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY", reasoning_effort: str = None, local_llm: bool = False, no_think: bool = False):
    """
    Stream LLM response, yielding complete sentences as they arrive.
    Yields (sentence, elapsed_since_start, ttft).
    """
    if local_llm:
        yield from stream_local_llm_sentences(text, model=model, no_think=no_think)
        return

    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} not set")

    client = OpenAI(api_key=api_key, base_url=base_url)

    t0 = perf_counter()
    buffer = ""
    ttft = None

    extra_kwargs = {}
    is_ollama = base_url and "localhost" in base_url or "127.0.0.1" in (base_url or "")
    system_prompt = LLM_SYSTEM_PROMPT
    if reasoning_effort and is_ollama:
        # Ollama doesn't support reasoning_effort API param; use /no_think tag
        system_prompt = LLM_SYSTEM_PROMPT + " /no_think"
    elif reasoning_effort:
        extra_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
        stream=True,
        **extra_kwargs,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            if ttft is None:
                ttft = perf_counter() - t0
            buffer += token

            # Yield on sentence boundary
            match = SENTENCE_ENDERS.search(buffer)
            if match:
                idx = match.end()
                while True:
                    m = SENTENCE_ENDERS.search(buffer, idx)
                    if m:
                        idx = m.end()
                    else:
                        break
                sentence = buffer[:idx].strip()
                buffer = buffer[idx:]
                if sentence:
                    yield sentence, perf_counter() - t0, ttft

    if buffer.strip():
        yield buffer.strip(), perf_counter() - t0, ttft


# ── Main benchmark ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E speech-to-speech benchmark")
    parser.add_argument("--audio", default=TTS_REF_AUDIO, help="Input audio file")
    parser.add_argument("--tts-model", default=TTS_MODEL_PATH, help="Path to Qwen3-TTS model")
    parser.add_argument("--ref-audio", default=TTS_REF_AUDIO)
    parser.add_argument("--ref-text", default=TTS_REF_TEXT)
    parser.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    parser.add_argument("--llm", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1", help="LLM API base URL")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY", help="Env var for LLM API key")
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"], help="Reasoning effort for thinking models")
    parser.add_argument("--local-llm", action="store_true", help="Use local transformers model instead of API")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load local LLM in 4-bit quantization")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking for Qwen3 models (local or API)")
    parser.add_argument("--no-unload", action="store_true", help="Don't unload Ollama model between runs (accept GPU contention)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="e2e_output.wav", help="Save final audio")
    args = parser.parse_args()

    print("=" * 60)
    print("  Loading models (one-time cost)")
    print("=" * 60)

    # Load STT
    print("Loading STT (Parakeet TDT 0.6B)...")
    stt_model = load_stt_model()

    # Warmup STT
    print("Warming up STT...")
    _ = run_stt(stt_model, args.audio)

    # Load TTS
    print(f"Loading TTS (Qwen3-TTS 0.6B, {args.attn})...")
    from faster_qwen3_tts import FasterQwen3TTS
    tts_model = FasterQwen3TTS.from_pretrained(
        args.tts_model, device="cuda", dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )

    # Warmup TTS (builds CUDA graphs)
    print("Warming up TTS...")
    for _ in tts_model.generate_voice_clone_streaming(
        text="Hello warmup sentence.", language="English",
        ref_audio=args.ref_audio, ref_text=args.ref_text,
        chunk_size=8, max_new_tokens=50,
    ):
        pass

    # Load & warmup LLM
    if args.local_llm:
        load_local_llm(args.llm, load_in_4bit=args.load_in_4bit)
    print(f"Warming up LLM ({args.llm})...")
    for _ in stream_llm_sentences("Hello", model=args.llm, base_url=args.llm_base_url, api_key_env=args.llm_api_key_env, reasoning_effort=args.reasoning_effort, local_llm=args.local_llm, no_think=args.no_think):
        pass

    print(f"\nRunning {args.runs} end-to-end benchmark(s)...\n")

    all_results = []

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n{'#' * 60}")
            print(f"  RUN {run_idx + 1}/{args.runs}")
            print(f"{'#' * 60}")

        cascade_start = perf_counter()

        # ── Stage 1: STT ────────────────────────────────────────────
        stt_text, stt_time = run_stt(stt_model, args.audio)
        print(f"\n📝 STT ({stt_time:.3f}s): \"{stt_text[:80]}\"")

        # ── Stage 2+3: LLM streaming → TTS ─────────────────────────
        llm_start = perf_counter()
        first_audio_time = None
        all_audio = []
        sr_out = 16000
        llm_ttft = None
        llm_first_sentence_time = None
        full_response = ""
        tts_start = None
        tts_ttfa = None

        for sentence, elapsed, ttft in stream_llm_sentences(stt_text, model=args.llm, base_url=args.llm_base_url, api_key_env=args.llm_api_key_env, reasoning_effort=args.reasoning_effort, local_llm=args.local_llm, no_think=args.no_think):
            if llm_ttft is None:
                llm_ttft = ttft
            if llm_first_sentence_time is None:
                llm_first_sentence_time = elapsed
                full_response = sentence
                print(f"🧠 LLM 1st sentence ({elapsed:.3f}s): \"{sentence}\"")

                # Free Ollama GPU memory before TTS if using Ollama (skip with --no-unload)
                if not args.local_llm and not args.no_unload and ("localhost" in args.llm_base_url or "127.0.0.1" in args.llm_base_url):
                    try:
                        import urllib.request
                        req = urllib.request.Request(
                            f"{args.llm_base_url.replace('/v1', '')}/api/generate",
                            data=json.dumps({"model": args.llm, "keep_alive": 0}).encode(),
                            headers={"Content-Type": "application/json"},
                        )
                        urllib.request.urlopen(req, timeout=5)
                        torch.cuda.synchronize()
                        time.sleep(0.1)  # brief pause for memory release
                    except Exception:
                        pass

                # Start TTS immediately on first sentence
                tts_start = perf_counter()
                for audio_chunk, sr, timing in tts_model.generate_voice_clone_streaming(
                    text=sentence, language="English",
                    ref_audio=args.ref_audio, ref_text=args.ref_text,
                    chunk_size=8, max_new_tokens=200,
                ):
                    if first_audio_time is None:
                        first_audio_time = perf_counter()
                        tts_ttfa = first_audio_time - tts_start
                    all_audio.append(audio_chunk)
                    sr_out = sr
            else:
                full_response += " " + sentence

        tts_done = perf_counter()

        # ── Results ─────────────────────────────────────────────────
        e2e_first_audio = first_audio_time - cascade_start if first_audio_time else None
        e2e_total = tts_done - cascade_start
        tts_total = tts_done - tts_start if tts_start else None

        audio_samples = sum(len(c) for c in all_audio)
        audio_duration = audio_samples / sr_out if audio_samples else 0
        tts_rtf = audio_duration / tts_total if tts_total and tts_total > 0 else 0

        print(f"🔊 TTS: {audio_duration:.2f}s audio, RTF {tts_rtf:.2f}x, TTFA {tts_ttfa:.3f}s" if tts_ttfa else "")

        print(f"\n{'=' * 60}")
        print(f"⚡ END-TO-END RESULTS")
        print(f"{'=' * 60}")
        print(f"")
        print(f"  ├─ STT:              {stt_time:.3f}s")
        print(f"  ├─ LLM TTFT:         {llm_ttft:.3f}s" if llm_ttft else "  ├─ LLM TTFT:         N/A")
        print(f"  ├─ LLM 1st sentence: {llm_first_sentence_time:.3f}s" if llm_first_sentence_time else "  ├─ LLM 1st sentence: N/A")
        print(f"  ├─ TTS TTFA:         {tts_ttfa:.3f}s" if tts_ttfa else "  ├─ TTS TTFA:         N/A")
        print(f"  └─ TTS total:        {tts_total:.3f}s (RTF {tts_rtf:.2f}x)" if tts_total else "  └─ TTS total:        N/A")
        print(f"")
        if e2e_first_audio:
            print(f"  🎯 E2E to first audio:  {e2e_first_audio:.3f}s")
        print(f"  📊 E2E total:           {e2e_total:.3f}s")
        print(f"{'=' * 60}")

        result = {
            "run": run_idx + 1,
            "stt_s": round(stt_time, 3),
            "stt_text": stt_text,
            "llm_ttft_s": round(llm_ttft, 3) if llm_ttft else None,
            "llm_first_sentence_s": round(llm_first_sentence_time, 3) if llm_first_sentence_time else None,
            "llm_response": full_response,
            "tts_ttfa_s": round(tts_ttfa, 3) if tts_ttfa else None,
            "tts_total_s": round(tts_total, 3) if tts_total else None,
            "tts_rtf": round(tts_rtf, 2),
            "tts_audio_s": round(audio_duration, 2),
            "e2e_first_audio_s": round(e2e_first_audio, 3) if e2e_first_audio else None,
            "e2e_total_s": round(e2e_total, 3),
            "attn": args.attn,
        }
        all_results.append(result)

        # Save audio from last run
        if all_audio and run_idx == args.runs - 1:
            audio_out = np.concatenate(all_audio)
            sf.write(args.output, audio_out, sr_out)
            print(f"\n💾 Audio saved: {args.output}")

    # Summary
    if len(all_results) > 1:
        e2e_times = [r["e2e_first_audio_s"] for r in all_results if r["e2e_first_audio_s"]]
        if e2e_times:
            print(f"\n{'=' * 60}")
            print(f"  SUMMARY ({len(e2e_times)} runs)")
            print(f"{'=' * 60}")
            print(f"  E2E first audio: {min(e2e_times):.3f}s (best) / {sum(e2e_times)/len(e2e_times):.3f}s (mean) / {max(e2e_times):.3f}s (worst)")
            stt_times = [r["stt_s"] for r in all_results]
            print(f"  STT:             {min(stt_times):.3f}s (best) / {sum(stt_times)/len(stt_times):.3f}s (mean)")
            ttfa_times = [r["tts_ttfa_s"] for r in all_results if r["tts_ttfa_s"]]
            if ttfa_times:
                print(f"  TTS TTFA:        {min(ttfa_times):.3f}s (best) / {sum(ttfa_times)/len(ttfa_times):.3f}s (mean)")

    with open("e2e_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: e2e_benchmark_results.json")


if __name__ == "__main__":
    main()
