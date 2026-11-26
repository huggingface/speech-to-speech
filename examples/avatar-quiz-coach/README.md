---
title: Viva Quiz Coach
emoji: 🎓
colorFrom: yellow
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 10080
short_description: A 3D avatar that researches any topic and quizzes you aloud
---

# Viva: a voice quiz coach with a talking avatar

Pick a topic, or type your own (“RLHF vs DPO”, “the French Revolution, focus
on the Terror”). Viva reads up on it, then quizzes you out loud. You answer in
your own words; she grades each answer, nudges you when you're halfway there,
explains what you missed, and keeps score with streaks, badges and a
leaderboard.

```
mic ─▶ speech-to-speech backend (VAD + STT) ─▶ transcript ─▶ Viva server (grade, score)
                                                                   │
avatar ◀─ lip-synced audio ◀─ speech-to-speech backend (TTS) ◀─ script to say
```

- **Ears and voice**: the browser speaks the OpenAI Realtime protocol to a
  [speech-to-speech](https://github.com/huggingface/speech-to-speech) server.
  `turn_detection.create_response: false` stops the backend from answering on
  its own; each line the coach says is an out-of-band `response.create`
  (`conversation: "none"`) carrying the exact script.
- **Brain**: `server.py` researches topics (Wikipedia, plus Google results when
  `SERPER_API_KEY` is set) into eight spoken-friendly questions with reference
  answers and key points, then grades every transcript with a fast model on
  Hugging Face Inference Providers. Points, streaks and badges are computed with
  plain rules in `coach/game.py`, so the leaderboard never relies on model arithmetic.
- **Face**: a [TalkingHead](https://github.com/met4citizen/TalkingHead) avatar.
  When the backend runs with `--enable_visemes`, its phoneme-based visemes drive
  the mouth; otherwise [HeadAudio](https://github.com/met4citizen/HeadAudio)
  derives visemes from the audio in the browser.

## Run locally

From this folder, with a speech-to-speech server on port 8765
(`speech-to-speech serve --enable_visemes`):

```bash
npm ci
curl -L -o static/avatars/avaturn.glb https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avaturn.glb
pip install -r requirements.txt
HF_TOKEN=hf_... SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime uvicorn server:app --port 7860
```

Open http://localhost:7860. Add `?fakemic=1` to test without a microphone
(`vivaFakeMic.play(url)` in the console speaks an audio file into it).
`python -m pytest tests` runs the scoring tests; `python scripts/bake_packs.py`
regenerates the featured topics.

## Configuration

| Env | What |
|---|---|
| `HF_TOKEN` | Inference Providers token for research and grading (required for new topics and grading) |
| `SPEECH_TO_SPEECH_URL` | Realtime WebSocket the browser dials directly |
| `LOAD_BALANCER_URL`, `LB_HF_TOKEN` | Use a speech-to-speech load balancer instead (its address stays server-side) |
| `SESSION_PROXY_URL` | Or piggyback another deployment's session API, e.g. `https://smolagents-hf-realtime-voice.hf.space/api` |
| `GRADER_MODELS`, `WRITER_MODELS` | Comma-separated fallback chains (defaults: Qwen3.8-27B on Cerebras for grading, GLM-5.3 for writing) |
| `COACH_VOICE` | TTS voice name sent to the backend (default `Serena`; empty keeps the backend's default) |
| `SERPER_API_KEY` | Adds Google results to research |
| `LEADERBOARD_REPO` | Dataset repo that mirrors the leaderboard so it survives restarts |
| `RESEARCH_PER_HOUR` | New topics per visitor per hour (default 8) |

The sample avatar comes from the TalkingHead repository and was made with
[Avaturn](https://avaturn.me), free for non-commercial use. The Docker build
downloads it from a pinned commit.
