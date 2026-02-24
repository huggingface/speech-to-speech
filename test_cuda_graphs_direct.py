#!/usr/bin/env python3
"""Test CUDA graphs using faster-qwen3-tts."""

import torch
import time
import soundfile as sf

from faster_qwen3_tts import FasterQwen3TTS

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
ref_audio = "ref_audio.wav"
ref_text = "I'm confused why some people have super short timelines."
text = "Hello from the CUDA graphs direct test. This should be real-time."

print("Loading model with CUDA graphs...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_PATH,
    device='cuda',
    dtype=torch.bfloat16,
    attn_implementation='eager',
    max_seq_len=2048,
)

print("\nWarmup (includes CUDA graph capture)...")
warmup_start = time.perf_counter()
audio_list, sr = model.generate_voice_clone(
    text=text[:20],
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=10,
)
warmup_time = time.perf_counter() - warmup_start
print(f"Warmup completed in {warmup_time:.2f}s")

print("\nGenerating audio...")
torch.cuda.synchronize()
start = time.perf_counter()

audio_list, sr = model.generate_voice_clone(
    text=text,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
    max_new_tokens=2048,
)

torch.cuda.synchronize()
gen_time = time.perf_counter() - start

audio = audio_list[0]
audio_duration = len(audio) / sr
rtf = audio_duration / gen_time
n_steps = int(audio_duration * 12)
ms_per_step = (gen_time * 1000) / n_steps if n_steps > 0 else 0

print(f"\n=== Results ===")
print(f"Steps: {n_steps}")
print(f"Audio duration: {audio_duration:.2f}s")
print(f"Generation time: {gen_time:.2f}s")
print(f"ms/step: {ms_per_step:.1f}ms")
print(f"RTF: {rtf:.3f}")
print(f"Real-time: {'✓' if rtf > 1.0 else '✗'}")

# Save audio
output_path = "test_cuda_graphs_output.wav"
sf.write(output_path, audio, sr)
print(f"\nSaved audio to {output_path}")
