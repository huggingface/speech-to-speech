"""Bake every model and recording the vibe check needs into the image.

A Jobs container is billed per second and starts with an empty cache, so anything
downloaded at run time is paid for on every run and is one Hub outage away from a
failed job. Pulling it at build time also means the image pins exactly what a run
will load, which is the point of a comparison harness.

Model ids are imported from the pipeline where they are module constants, so this
cannot silently drift from what the server actually loads.
"""

from __future__ import annotations

import sys

from huggingface_hub import hf_hub_download, snapshot_download

from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.evals.big_bench_audio.dataset import load_subset, resolve_audio
from speech_to_speech.VAD.smart_turn import MODEL_FILENAME, MODEL_REPO_ID

# Chosen by ParakeetTDTSTTHandler on CUDA/CPU; see STT/parakeet_tdt_handler.py.
PARAKEET_REPO_ID = "nvidia/parakeet-tdt-0.6b-v3"


def fetch_speech_models() -> None:
    tts_repo = Qwen3TTSHandlerArguments().qwen3_tts_model_name
    for repo_id in (PARAKEET_REPO_ID, tts_repo):
        print(f"prefetch model {repo_id}", flush=True)
        snapshot_download(repo_id)

    print(f"prefetch model {MODEL_REPO_ID}/{MODEL_FILENAME}", flush=True)
    hf_hub_download(MODEL_REPO_ID, MODEL_FILENAME)


def fetch_silero_vad() -> None:
    """Silero comes from torch.hub, not the Hub; see VAD/vad_handler.py."""
    import torch

    print("prefetch model snakers4/silero-vad", flush=True)
    torch.hub.load("snakers4/silero-vad:master", "silero_vad", trust_repo=True, skip_validation=True)


def fetch_nltk() -> None:
    import nltk

    for package in ("punkt_tab", "averaged_perceptron_tagger_eng"):
        print(f"prefetch nltk {package}", flush=True)
        nltk.download(package)


def fetch_eval_audio(subset_names: list[str]) -> None:
    for name in subset_names:
        subset = load_subset(name)
        print(f"prefetch audio for subset {name} ({len(subset)} recordings)", flush=True)
        for item in subset.items:
            resolve_audio(item, subset)


def main(argv: list[str]) -> int:
    fetch_speech_models()
    fetch_silero_vad()
    fetch_nltk()
    fetch_eval_audio(argv or ["vibe"])
    print("prefetch complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
