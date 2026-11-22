# Long-term memory (optional)

The Realtime server can recall what it knows about the user before each reply, and save what the user asks it to
remember, through the [AgentMemory](https://github.com/joslat/agent-memory-dotnet) native library
(`agentmemory-native` on PyPI).

Memory is off unless `S2S_MEMORY_CONFIG` is set. With it unset, the server does not import `agentmemory-native`,
builds every request exactly as before, adds no route and adds nothing to `/v1/realtime/capabilities`.

## Install

```bash
pip install "speech-to-speech[memory]"
```

`agentmemory-native` ships wheels for Linux x86-64 and ARM64 (glibc 2.34 or newer), macOS on Apple silicon and
Windows x64. It needs a Neo4j 5.26 store with the AgentMemory schema and an OpenAI-compatible embeddings endpoint;
the writing profiles also need an OpenAI-compatible chat model.

If `S2S_MEMORY_CONFIG` is set and the package is missing, or the library refuses the configuration, the server stops
at startup and logs why. It never runs with memory silently off.

## Configure

`S2S_MEMORY_CONFIG` is the path to an AgentMemory configuration file (JSON). A read-only example:

```json
{
  "schemaVersion": 1,
  "profile": "read_only",
  "owner": {"mode": "fixed", "id": "user-1"},
  "store": {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "..."},
  "embedding": {"dimensions": 1024, "model": "bge-m3",
                "endpoint": {"url": "http://localhost:11434/v1/embeddings"}}
}
```

The profiles and every field are described in the library's
[contract](https://github.com/joslat/agent-memory-dotnet/blob/main/native/CONTRACT.md). One process serves one owner.

| Variable | Default | Meaning |
| --- | --- | --- |
| `S2S_MEMORY_CONFIG` | unset (off) | Path to the AgentMemory configuration |
| `S2S_MEMORY_DEADLINE_MS` | `450` | How long a request may wait for its recall, counted from when the request is prepared |
| `S2S_MEMORY_MAX_FACTS` | `10` | Facts per recall |
| `S2S_MEMORY_MAX_PREFERENCES` | `5` | Preferences per recall |
| `S2S_MEMORY_MIN_SCORE` | `0.70` | Similarity cut-off |
| `S2S_MEMORY_MAX_ENTITIES` | `10` | People and things per recall |
| `S2S_MEMORY_MAX_RELATIONSHIPS` | `10` | Relations between them per recall |
| `S2S_MEMORY_MAX_MESSAGES` | `0` | Earlier messages of the same conversation per recall |
| `S2S_MEMORY_TIME_WINDOWS` | `1` | `0` turns off recalling the facts of a time the question names ("yesterday") |
| `S2S_MEMORY_VOICE_REMEMBER` | `1` | `0` turns off remembering on a spoken request (writing profiles) |
| `S2S_MEMORY_OBSERVE` | `0` | `1` with the `learning` profile: also give the writer what the user tells unasked |
| `S2S_MEMORY_DEPLOYMENT` | `s2s` | Prefix of the source ids of saved utterances |
| `S2S_MEMORY_CONTROL_TOKEN_FILE` | unset | File holding the bearer token for `POST /v1/memory/erase` |

## What happens on a turn

1. When a user turn becomes a model request, a recall for its text starts on a background thread. The request
   carries it.
2. The model handler (`chat-completions` or `responses-api`) waits for the recall at most until the deadline. It then
   adds the recalled memories, with today's date, as a labelled text part of the current user message, in its own
   copy of the conversation. The stored conversation, history and compaction never see it.
3. A late or failed recall, a turn whose text changed since, or a missing user message means the request goes without
   memory. A recall that fails starts a background warm-up loop until the store answers again.

Memories recalled earlier in a session are carried to its later turns (at most 40 per turn). When a client connects,
a few quiet recalls fill what the conversation starts with. Speculative tool follow-up prefetch is off while memory is
on.

## Remember and forget by voice

With the `explicit_write` profile, a final spoken utterance that starts by asking ("remember that ...", "please
remember ...", "don't forget ...") is saved from the speech recognizer's words, never the model's. That turn's request
tells the model only what the library's receipt supports: saving, saved, not saved or not confirmed. A later outcome is
told once, on the next turn. "Forget what I just told you" closes what this connection last saved.

With the `learning` profile and `S2S_MEMORY_OBSERVE=1`, statements the user makes unasked are also given to the
library's writer, which decides what to keep. The model is not told about these writes.

## Endpoints

With memory on, `GET /v1/realtime/capabilities` gains a `memory` section: enabled, the profile, `ready` or
`recovering`, the deadline and, for writing profiles, how many writes are saving. It never includes the owner, the
store, the configuration or anything remembered.

With a writing profile and `S2S_MEMORY_CONTROL_TOKEN_FILE` set, `POST /v1/memory/erase` with
`Authorization: Bearer <token>` and the body `{"confirm": true}` erases everything stored for the configured owner.
Without the token file it answers 404.
