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
on the Terror”). Your coach (Max, Ada or Reachy) reads up on it, then quizzes you out
loud. You answer in your own words. A wrong answer or “I don't know” gets a
clue, not the answer: up to three tries, each clue more specific, and the
answer is explained only after the last try or when you say “skip”. Streaks,
badges and a leaderboard keep score.

```
mic ─▶ speech-to-speech backend: VAD + STT ─▶ LLM (the coach) ─▶ TTS ─▶ lip-synced avatar
                                                   │  ▲
                              check_answer, give_hint, skip_question, end_quiz
                                                   ▼  │
                                  coach server: grade, clue, score (via the browser)
```

- **Conversation**: the browser speaks the OpenAI Realtime protocol to a
  [speech-to-speech](https://github.com/huggingface/speech-to-speech) server,
  whose LLM is the coach: it answers on its own after each turn, chats,
  repeats or explains a question, and stops when you talk over it (server VAD
  with `interrupt_response`; the browser flushes its audio and truncates the
  reply to what you heard). The voice plays through Web Audio, which the
  browser's echo canceller doesn't hear, so during the coach's first two
  seconds of speech the mic is muted and its level is compared with the
  coach's: on headphones barge-in turns on; if the mic tracks the coach's
  voice (loudspeakers), the coach stops listening while it talks instead of
  answering itself. Session instructions (`coach/voice.py`) carry the
  quiz state but never the answers. When you attempt one, the LLM calls
  `check_answer`; the browser runs the grader on your actual transcript and
  returns the verdict and what to say (a clue, or the result and the next
  question). While the grader works the coach says one of 40 short
  fillers ("let's see", "hang on a second"…), recorded in its voice during
  the session and picked least-played first, so they rarely repeat. When a pause splits an answer, the
  coach answers an unfinished piece with just "mm-hm", and if the
  student goes on before hearing the clue for the first half, that grade is
  taken back: the try doesn't count and the halves are graded together.
  In 68 scripted text turns (six quizzes) on the dev stack's Gemma 4 31B, the
  coach picked the right tool, or none for small talk, "repeat that" and
  trailing off, every time, leaked no answers, and read 24 of 24 next
  questions word for word. In one voice test it wrote a call out as
  text instead; such a reply is cut from playback and history and retried
  with a tool call required.
- **Brain**: `server.py` researches topics into eight spoken-friendly
  questions. A GitHub link or a paper (an arXiv or Hugging Face Papers link, or
  a paper named by title) is read first-hand: the README, design docs and an
  outline of the main source files, or the paper's full text. Other topics use
  Wikipedia, plus Google results when `SERPER_API_KEY` is set. The writer lists
  the key concepts first and writes spares, so repeats are dropped. Questions come with reference
  answers and key points (marked as required when the question asks for them,
  context otherwise), then grades answers and writes clues with a fast model
  on Hugging Face Inference Providers. Grading starts as soon as the
  transcript arrives (or from the backend's transcript snapshot while it waits
  out the reopen grace), so it is often done when the coach calls for it. A check rejects clues that
  contain answer phrases the student hasn't said yet. Points, streaks and badges
  are computed with plain rules in `coach/game.py`.
- **Face**: a [TalkingHead](https://github.com/met4citizen/TalkingHead) avatar (Max),
  a photoreal Gaussian-splat head (Ada) or Pollen Robotics'
  [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) (Reachy).
  When the backend runs with `--enable_visemes`, its phoneme-based visemes drive
  the mouth, merged into held, blended shapes and scheduled against the audio
  player; otherwise [HeadAudio](https://github.com/met4citizen/HeadAudio)
  derives visemes from the audio in the browser. Reachy Mini has no mouth, so
  the same timing moves its head, as the real robot does when it talks: it
  rises on vowels and settles on closures, and the antennas flick on
  syllables. Its reactions are emotions recorded on the real robot.

## Run locally

From this folder, with a speech-to-speech server on port 8765
(`speech-to-speech serve --enable_visemes`):

```bash
npm ci
curl -L -o static/avatars/avatarsdk.glb https://raw.githubusercontent.com/met4citizen/TalkingHead/1563de184c19b310de4b636664d5835e2c74dbba/avatars/avatarsdk.glb
curl -L -o static/avatars/nyx.zip https://cdn.jsdelivr.net/npm/@myned-ai/avatar-chat-widget@1.15.0/public/asset/nyx.zip
curl -L -o static/avatars/reachy_mini.glb https://raw.githubusercontent.com/andimarafioti/agentic-days-keynote/a663a678fc03b69a6b920d089f4a9bdfef10d3db/assets/reachy_mini_viz.glb
python scripts/bake_reachy_moves.py
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
regenerates the featured topics. http://localhost:7860/static/lab.html?coach=reachy
drives a coach without a voice session: it speaks the clips in `static/_test`
(any 24 kHz mono WAVs) through the same calls as the backend and fires the
reactions.

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
| `COACH_VOICE` | Overrides the TTS voice (by default Max and Reachy use `en-arthur_m` and Ada `en-maya_f`, voices of the hosted stack; a Qwen3-TTS CustomVoice backend needs e.g. `Aiden` / `Serena`) |
| `SERPER_API_KEY` | Adds Google results to research |
| `LOG_TURNS` | `1` logs each graded turn (the student's words and the grader's decision) for debugging |
| `GITHUB_TOKEN` | Optional, for reading GitHub repositories past the anonymous API limit (60 requests an hour per IP) |
| `LEADERBOARD_REPO` | Dataset repo that mirrors the leaderboard so it survives restarts |
| `RESEARCH_PER_HOUR` | New topics per visitor per hour (default 8) |

Max is TalkingHead's [Avatar SDK](https://avatarsdk.com) sample (free for
non-commercial use). Ada is "Nyx", a photoreal Gaussian-splat head made with
[LAM](https://github.com/aigc3d/LAM) and shipped (MIT) with
[avatar-chat-widget](https://github.com/myned-ai/avatar-chat-widget); it is drawn by
[gsplat-flame-avatar-renderer](https://github.com/myned-ai/gsplat-flame-avatar-renderer)
and driven with ARKit blendshapes, head and shoulders only. Reachy is
[Reachy Mini](https://github.com/pollen-robotics/reachy_mini), Pollen Robotics'
open-source desk robot: the model is from
[reachy_mini_blender](https://github.com/pollen-robotics/reachy_mini_blender)
(Apache-2.0, by Clément Plays; the 1.3 MB Draco export from Fabien Danieau's
[Reachy Mini talk](https://github.com/FabienDanieau/reachy_mini_intro)), posed by
the rig from Pollen's
[reachy-mini-yo-its-your-boy](https://huggingface.co/spaces/pollen-robotics/reachy-mini-yo-its-your-boy)
Space (Apache-2.0, `static/js/reachy-rig.js`), and its reactions come from the
[emotions library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
(Apache-2.0). The Docker build downloads the models from pinned, checksummed
URLs and bakes the emotions.
