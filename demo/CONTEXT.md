# Context glossary

Canonical terms for this space. A glossary, not a spec — it defines what words mean,
not how anything is built. Keep design/implementation detail in `DESIGN.md` and the
code.

## Speech-to-speech demo
The product: a voice conversation you have with a model by tapping the orb and
talking. "The demo" and "the space" refer to this same thing. It runs on Hugging
Face's open `speech-to-speech` backend.

## The pipeline
The ordered path a turn travels, from your voice to the orb's reply. Order is
meaningful — each stage consumes the previous one's output:

`you speak → VAD → STT → VLM → TTS → orb replies`

- **VAD** — voice activity detection. Decides *when* you are speaking, so the system
  knows a turn has started and ended. Model: silero-vad.
- **STT** — speech to text. Transcribes your speech into words. Model:
  nvidia/parakeet-tdt-1.1b.
- **VLM** — the vision-language model that composes the reply. Served via Cerebras.
  Model: google/gemma-4-31B-it.
- **TTS** — text to speech. Speaks the reply back in the chosen voice. Model:
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice.

## Builder
A Hugging Face user credited with making the space, shown by HF username. Current
builders: tfrere, A-Mahla and andito. Distinct from the *models'* authors (nvidia, google,
Qwen, snakers4), who are credited per pipeline stage.

## Powered by
The infrastructure running the pipeline, named in the about panel: Hugging Face
Inference Endpoints (hosting) and Cerebras (LLM inference). Distinct from "built by"
(the people) and from the model authors (who trained each model).

## Tool
A function the model can call mid-conversation to do something the pipeline
can't do on its own (look something up, look through the camera). Tools are
declared to the backend in the session config; the model decides when to call
one. Distinct from the *pipeline stages* (VAD/STT/VLM/TTS), which always run.

## Tool executor
The client-side component that runs a tool when the model calls it and returns
the result to the backend, so the model can speak the answer. It is the missing
half of the round-trip: the backend already emits the call, the executor runs it
and replies. Distinct from the *tool* itself (the thing being run).

## Web search tool
A tool that looks something up on the web for the model. The model calls it with
a query; the tool executor forwards the query to the search proxy and returns the
results as the tool result. Activates only when a search key is available.

## Camera snapshot tool
A tool that lets the model look through the user's webcam. While enabled, a live
self-view is shown in the page (bottom-left); when the model calls the tool, the
executor captures a frame and sends it to the model as an image so the VLM can
see it. Distinct from the *preview* (what the user sees) and the *snapshot* (the
single frame sent to the model).

## Search proxy
The same-origin server route (`/search`) that holds the search key and calls the
external search provider on the client's behalf, so the key never reaches the
browser. Lives in the same container as the page. Distinct from the *s2s backend*
(the separate load-balanced speech-to-speech service).

## Tools panel
The dialog opened from the "Tools" button in the top-right, holding one switch per
tool (and the web-search key status). Turning a switch on/off declares or removes
that tool on the live session. Distinct from *Settings* (connection, voice,
instructions) and the *About panel* (project info).

## Identity block
The top-left corner of the topbar (replacing the old wordmark): the demo name, a
one-line blurb, and the "powered by" / "built by" credits, shown directly rather
than hidden behind a click.

## About panel
The popup opened from the (i) icon to the right of the identity block. Holds the
general introduction to the speech-to-speech project (with a repo link) and the
pipeline. Identity itself now lives in the corner, not here.

## Queue
The line of users waiting for a free conversation slot when every compute is
busy. You join the queue instead of being turned away; you leave it by reaching
the front or by giving up. Distinct from a *session* (an actual live
conversation) — being in the queue is not yet talking, and time spent waiting
never counts against your usage limit.

## Ticket
Your held place in the queue. Created when you join, it is what the demo checks
to tell you your position and to notice if you have left. A ticket is not a
session: it only promises a spot in line, not a compute.

## Position
How many people are ahead of you in the queue, shown while you wait ("You're #3
in line"). It only ever counts down. Distinct from an *estimated wait* — the
demo shows position, never a time, because wait time is unpredictable.

## Claim
The moment you reach the front and a free slot becomes yours — the queue hands
off to a real session and the conversation begins. This is also the point where
your usage limit first starts to matter (never while waiting).

## At capacity
The state where the queue itself is full, so new users can't even join the line
and are asked to try again shortly. Distinct from simply *busy* (all computes
taken but the queue still has room to wait in).
