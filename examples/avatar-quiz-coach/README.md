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
short_description: An avatar coach that researches any topic and quizzes you aloud
---

# Viva: a voice quiz coach with a talking avatar

Pick a topic, or type your own (“RLHF vs DPO”, “the French Revolution, focus
on the Terror”). Your coach (Max or Ada) reads up on it, then quizzes you out
loud. You answer in your own words. A wrong answer or “I don't know” gets a
clue, not the answer: up to three tries, each clue more specific, and the
answer is explained only after the last try or when you say “skip”. Streaks,
badges and a leaderboard keep score.

```
mic ─▶ speech-to-speech backend (VAD + STT) ─▶ transcript ─▶ coach server (grade, clue, score)
                                                                   │
avatar ◀─ lip-synced audio ◀─ speech-to-speech backend (TTS) ◀─ script to say
```

- **Ears and voice**: the browser speaks the OpenAI Realtime protocol to a
  [speech-to-speech](https://github.com/huggingface/speech-to-speech) server.
  `turn_detection.create_response: false` stops the backend from answering on
  its own; each line the coach says is an out-of-band `response.create`
  (`conversation: "none"`) carrying the exact script. The backend has no
  text-to-speech-only path, so its LLM reads the line aloud: lines are sent at
  most two sentences at a time with few-shot read-aloud examples, which kept
  24 of 24 test lines word for word on Qwen3.5-4B and Gemma 4 31B (a plain
  "repeat this" prompt made the model answer the quiz questions itself).
- **Brain**: `server.py` researches topics (Wikipedia, plus Google results when
  `SERPER_API_KEY` is set) into eight spoken-friendly questions with reference
  answers and key points (marked as required when the question asks for them,
  context otherwise), then grades every transcript and writes clues with a
  fast model on Hugging Face Inference Providers. Grading starts speculatively
  from the backend's transcript snapshot as soon as the student pauses, so it
  runs during the backend's reopen grace instead of after it. A check rejects clues that
  contain answer phrases the student hasn't said yet. Points, streaks and badges
  are computed with plain rules in `coach/game.py`.
- **Face**: a [TalkingHead](https://github.com/met4citizen/TalkingHead) avatar (Max)
  or a photoreal Gaussian-splat head (Ada).
  When the backend runs with `--enable_visemes`, its phoneme-based visemes drive
  the mouth, merged into held, blended shapes and scheduled against the audio
  player; otherwise [HeadAudio](https://github.com/met4citizen/HeadAudio)
  derives visemes from the audio in the browser.

## Run locally

From this folder, with a speech-to-speech server on port 8765
(`speech-to-speech serve --enable_visemes`):

```bash
npm ci
curl -L -o static/avatars/avatarsdk.glb https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avatarsdk.glb
curl -L -o static/avatars/nyx.zip https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/public/asset/nyx.zip
pip install -r requirements.txt
HF_TOKEN=hf_... SPEECH_TO_SPEECH_URL=ws://localhost:8765/v1/realtime uvicorn server:app --port 7860
```

A speech-to-speech server running entirely on a laptop is slow here: its LLM
re-reads every coach line before text-to-speech can start (1.6 to 6 seconds a
line on an Apple Silicon Mac that also renders the avatar). For realistic
latency, point it at a hosted LLM (see the main README's LLM backends) or use a
deployed stack through its load balancer:

```bash
LOAD_BALANCER_URL=https://<lb>.endpoints.huggingface.cloud LB_HF_TOKEN=$HF_TOKEN LB_USER_TOKEN=$HF_TOKEN \
HF_TOKEN=hf_... uvicorn server:app --port 7860
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
| `LB_USER_TOKEN` | Local development only: stands in for the signed-in visitor's token when the load balancer requires a verified HF token. Never set it on a public Space. |
| `SESSION_PROXY_URL` | Or piggyback another deployment's session API, e.g. `https://smolagents-hf-realtime-voice.hf.space/api` |
| `GRADER_BASE_URL`, `GRADER_MODEL` | Grade on the speech backend's own model through any OpenAI-compatible endpoint (e.g. the deployment's Gemma 4 31B); `GRADER_REASONING_EFFORT` (default `none`) and `GRADER_API_KEY` (default `HF_TOKEN`) go with it. The hosted chain below stays as fallback. |
| `GRADER_MODELS`, `WRITER_MODELS` | Comma-separated Inference Providers fallback chains (defaults: Qwen3.8-27B on Cerebras with reasoning off for grading, GLM-5.3 for writing; `model@effort` sets a model's reasoning effort) |
| `COACH_VOICE` | Overrides the TTS voice (by default Max uses `en-arthur_m` and Ada `en-maya_f`, voices of the hosted stack; a Qwen3-TTS CustomVoice backend needs e.g. `Aiden` / `Serena`) |
| `SERPER_API_KEY` | Adds Google results to research |
| `LEADERBOARD_REPO` | Dataset repo that mirrors the leaderboard so it survives restarts |
| `RESEARCH_PER_HOUR` | New topics per visitor per hour (default 8) |

Max is TalkingHead's [Avatar SDK](https://avatarsdk.com) sample (free for
non-commercial use). Ada is "Nyx", a photoreal Gaussian-splat head made with
[LAM](https://github.com/aigc3d/LAM) and shipped (MIT) with
[avatar-chat-widget](https://github.com/myned-ai/avatar-chat-widget); it is drawn by
[gsplat-flame-avatar-renderer](https://github.com/myned-ai/gsplat-flame-avatar-renderer)
and driven with ARKit blendshapes, head and shoulders only. The Docker build
downloads both from pinned, checksummed URLs.
