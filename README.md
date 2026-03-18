<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="600"/>
</div>

# Speech To Speech MVP: Local Norwegian-to-English live translation

This repository is currently tuned for one local-first use case:

- Input: Norwegian speech from the PC microphone
- Output: English speech on the local speakers
- Runtime: Local desktop only for this MVP

Web, mobile, and broader multilingual workflows are intentionally out of scope for this first task.

## What the pipeline does

The active path is:

1. Voice activity detection segments microphone audio.
2. STT listens for Norwegian speech.
3. The language model rewrites the utterance as natural English.
4. TTS speaks the English translation through the local speakers.

Default behavior now favors a simple local setup:

- `--mode local` is the default.
- Whisper-based STT defaults to Norwegian input.
- The language model defaults to English-only translation output.
- Non-macOS defaults to Pocket TTS.

## Setup

Clone the repository:

```bash
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
```

Install dependencies with `uv`:

```bash
uv sync
```

On macOS, if you plan to use the default Melo TTS path, install UniDic once:

```bash
uv run python -m unidic download
```

Audio prerequisites:

- Allow microphone access for your terminal or editor.
- Make sure your default input and output devices are set correctly in the OS.
- The first run will download model weights, so expect a slower startup.

## Microphone test

Before running translation, verify that the PC microphone works:

```bash
uv run python microphone_test.py
```

This prints a live level meter from the default microphone.

If you want to hear the microphone routed to the speakers as well:

```bash
uv run python microphone_test.py --monitor true
```

Use headphones if you enable monitoring to avoid feedback.

## Run the local translator

### Linux or Windows workstation

Run the local pipeline with the current defaults:

```bash
uv run python s2s_pipeline.py
```

If you want a more explicit CPU-only command:

```bash
uv run python s2s_pipeline.py \
  --device cpu \
  --tts pocket \
  --pocket_tts_device cpu
```

### Apple Silicon Mac

Use the macOS shortcut configuration:

```bash
uv run python s2s_pipeline.py --local_mac_optimal_settings
```

That keeps the same MVP behavior, but swaps in the Apple Silicon friendly backends.

## Useful flags

The MVP works without extra flags, but these are the most useful overrides:

- `--device cpu`, `--device cuda`, `--device mps`
- `--stt whisper`, `--stt whisper-mlx`, `--stt parakeet-tdt`
- `--tts pocket`, `--tts melo`
- `--log_level debug`

Inspect the full CLI surface with:

```bash
uv run python s2s_pipeline.py -h
```

## Notes

- `listen_and_play.py` and the socket or websocket code are still present in the repository, but they are not part of this MVP flow.
- Pocket TTS remains the simplest default on non-macOS.
- If you want same-language transcription instead of translation, override Whisper with `--stt_gen_task transcribe`.

## Troubleshooting

No input on the microphone meter:

- Check OS microphone permissions.
- Check the default recording device.
- Try `uv run python microphone_test.py --input_device <index>` after listing devices with:

```bash
uv run python -c "import sounddevice as sd; print(sd.query_devices())"
```

No audio output from the translator:

- Confirm the local speaker device works outside the app.
- On Linux CPU, retry with `--tts pocket --pocket_tts_device cpu`.
- On macOS, make sure `uv run python -m unidic download` has already been run.

## Citations

### Silero VAD
```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```

### Distil-Whisper
```bibtex
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling},
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Parler-TTS
```bibtex
@misc{lacombe-etal-2024-parler-tts,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Parler-TTS},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/parler-tts}}
}
```
