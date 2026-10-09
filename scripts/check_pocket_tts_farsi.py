"""Run the Pocket TTS Farsi handler with real weights and save WAVs for listening.

Requires the Pocket TTS fork documented in README.md. Downloads both models and
the reference recording on first use. This checks synthesis, not pronunciation.
"""

import argparse
import json
from pathlib import Path
from queue import Queue
from threading import Event
from time import perf_counter

import numpy as np
import torch
from scipy.io import wavfile

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse, TTSInput
from speech_to_speech.TTS.pocket_tts_handler import PocketTTSHandler

MODEL = "mehdi-hf/pocket-tts-farsi-v2"
REFERENCE = f"hf://{MODEL}/samples/prompt_short_sentence.wav"
CASES = {
    "greeting": "سلام، حال شما چطور است؟",
    "numbers": "تا سال ۲۰۳۰ تغییر دهد.",
    "negative_numbers": "دمای هوا −۵ درجه است.",
    "sentences": "مادر کتاب را روی میز اتاق گذاشت. پنجره را باز کرد.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=REFERENCE, help="Reference audio file or URL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--eos-threshold", type=float, default=-2.0)
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=(16000, 24000),
        default=24000,
        help="WAV output sample rate. Default 24000 preserves the model's native audio.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pocket-tts-farsi-check"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    handler = PocketTTSHandler.__new__(PocketTTSHandler)
    handler.queue_out = Queue()
    start = perf_counter()
    handler.setup(
        Event(),
        model_name=MODEL,
        voice=args.voice,
        device=args.device,
        sample_rate=args.sample_rate,
        temperature=args.temperature,
        eos_threshold=args.eos_threshold,
    )
    assert handler.phonemizer is not None
    report = {
        "model": MODEL,
        "voice": args.voice,
        "device": handler.device,
        "setup_seconds": perf_counter() - start,
        "sample_rate": handler.sample_rate,
        "seed": args.seed,
        "temperature": handler.model.temp,
        "eos_threshold": handler.model.eos_threshold,
        "frames_after_eos": 0,
        "cases": [],
    }
    for name, text in CASES.items():
        torch.manual_seed(args.seed)
        phonemes = handler.phonemizer(text)
        if not phonemes or any("\u0600" <= char <= "\u06ff" for char in phonemes):
            raise RuntimeError(f"{name}: G2P did not produce romanized phonemes")
        if name == "negative_numbers":
            assert phonemes == handler.phonemizer("دمای هوا منفی پنج درجه است.")
            assert phonemes != handler.phonemizer("دمای هوا پنج درجه است.")
        start = perf_counter()
        first_audio_seconds = None
        blocks = []
        for block in handler.process(TTSInput(text=text, language_code="fa")):
            if not isinstance(block, np.ndarray) or block.dtype != np.int16 or block.shape != (handler.blocksize,):
                raise RuntimeError(f"{name}: invalid pipeline audio block")
            if first_audio_seconds is None:
                first_audio_seconds = perf_counter() - start
            blocks.append(block)
        if not handler.queue_out.empty():
            raise RuntimeError(f"{name}: {handler.queue_out.get_nowait()}")
        elapsed = perf_counter() - start
        if not blocks:
            raise RuntimeError(f"{name}: synthesis returned no audio")
        audio = np.concatenate(blocks)
        duration = len(audio) / handler.sample_rate
        rms = float(np.sqrt(np.mean((audio.astype(np.float64) / 32768) ** 2)))
        if duration < 0.5 or rms < 0.001:
            raise RuntimeError(f"{name}: synthesis was too short or silent, {duration=}, {rms=}")
        if list(handler.process(EndOfResponse())) != [AUDIO_RESPONSE_DONE]:
            raise RuntimeError("EndOfResponse did not emit the audio completion marker")
        output = args.output_dir / f"{name}.wav"
        wavfile.write(output, handler.sample_rate, audio)
        result = {
            "name": name,
            "text": text,
            "phonemes": phonemes,
            "wav": str(output),
            "blocks": len(blocks),
            "duration_seconds": duration,
            "synthesis_seconds": elapsed,
            "first_audio_seconds": first_audio_seconds,
            "rms": rms,
        }
        report["cases"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Passed. Audio and report saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
