"""Optional long-term memory for the realtime server, through the AgentMemory native library.

Off unless ``S2S_MEMORY_CONFIG`` names an AgentMemory native configuration (JSON). It needs the optional
``agentmemory-native`` package (``pip install "speech-to-speech[memory]"``), which is imported only when memory is on.
With memory off nothing here imports that package, and requests are built exactly as before.

What it does when on (profile "read_only"):
- before generation: when a user turn becomes a model request, a recall for that turn's text starts at once on a
  bounded background thread; the request carries it (``GenerateResponseRequest.memory``);
- at generation: the model handler waits for the recall at most until its deadline, then writes the recalled facts and
  preferences into the anchored user item of its own deep copy of the conversation, never into the stored chat;
- a late result, a superseded turn (the item's text changed) or a missing anchor means the request goes without memory.

With the library's "explicit_write" profile it also remembers on request:
- a final spoken utterance that starts by asking ("remember that ...", "please remember ...", "don't forget ...") is
  remembered from the microphone's own words through the library's durable journal, under a stable source id (this
  connection, item and revision), whether or not the turn gets a response;
- that turn's request carries a memory-status line saying only what the write's receipt supports: saving (not
  confirmed), saved, not saved, or not confirmed; an outcome that arrives later is told once, on the next turn;
- while a write is pending or uncertain the library pauses recall, and turns go without memory (not an error).

With the "learning" profile and ``S2S_MEMORY_OBSERVE=1`` it also gives the writer what the owner tells unasked.
See README.md in this directory for the configuration.
"""

from .provider import (
    MemoryPreparation,
    WriteTicket,
    memory_erase,
    memory_provider,
    memory_status,
    remember_intent,
    render_memory,
    reset_memory_provider,
)

__all__ = [
    "MemoryPreparation",
    "WriteTicket",
    "memory_erase",
    "memory_provider",
    "memory_status",
    "remember_intent",
    "render_memory",
    "reset_memory_provider",
]
