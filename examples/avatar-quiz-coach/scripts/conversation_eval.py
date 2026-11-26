"""Text-only check of the conversational coach against a speech-to-speech server.

A scripted student talks to the backend's LLM with the coach's instructions and
tools; tool calls run the real grader. Reports whether the right tool (or none)
was called, answers the coach leaked, and next questions read word for word.

    python scripts/conversation_eval.py ws://localhost:8765/v1/realtime packs/transformers.json 2
    python scripts/conversation_eval.py https://<load balancer> packs/diffusion.json   # needs HF_TOKEN

Needs `websockets` (not an app dependency).
"""

import asyncio
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coach import game as g  # noqa: E402
from coach import voice  # noqa: E402

TARGET, PACK = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1


def connect_url():
    if TARGET.startswith("ws"):
        return TARGET
    tok = os.environ["HF_TOKEN"]
    req = urllib.request.Request(
        f"{TARGET}/session",
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {tok}",
            "X-Reachy-Mini-Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
        },
    )
    return json.load(urllib.request.urlopen(req, timeout=30))["connect_url"]


def correct_answer(q):
    pts = [q["key_points"][i] for i in q.get("required", range(len(q["key_points"])))]
    return "I think " + " and ".join(pts)


# (label, student line or callable(question) -> line, expected tool or None)
SCRIPT = [
    ("ready", "Hi! Yeah, I'm ready, let's go.", None),
    ("repeat", "Sorry, could you repeat the question?", None),
    ("correct", correct_answer, "check_answer"),
    ("scope", "What kind of answer are you looking for here?", None),
    ("wrong", "Hmm, I think it has something to do with convolutions and pooling layers.", "check_answer"),
    ("tell-me", "Just tell me the answer, I have no clue.", "skip_question"),
    ("hint", "Can I get a hint for this one?", "give_hint"),
    ("dunno", "I honestly have no idea.", "check_answer"),
    ("correct2", correct_answer, "check_answer"),
    ("smalltalk", "By the way, how's your day going?", None),
    ("trailing", "Hmm, I think it's about the", None),
    ("stop", "Okay, I think I'll stop here for today.", "end_quiz"),
]


async def collect(ws):
    """Events of one response: (text, [function calls], status, first-token latency)."""
    text, calls, t0, first = "", [], time.perf_counter(), None
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 90))
        t = ev["type"]
        if t in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            first = first or time.perf_counter() - t0
            text += ev.get("delta", "")
        elif t == "response.function_call_arguments.done":
            first = first or time.perf_counter() - t0
            calls.append({"name": ev["name"], "call_id": ev["call_id"], "arguments": ev.get("arguments") or "{}"})
        elif t == "error":
            print("   ! error:", ev["error"].get("message"))
            if "active response" not in ev["error"].get("message", ""):
                return text, calls, "error", first
        elif t == "response.done":
            return text, calls, ev["response"]["status"], first


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def verbatim(question, text):
    q, t = " ".join(words(question)), " ".join(words(text))
    return q in t


async def run(rep):
    pack = json.load(open(PACK))
    game = g.Game(
        id="t",
        pack=pack,
        player="Sam",
        verified=False,
        practice=False,
        question_ids=[q["id"] for q in pack["questions"]],
        coach="Ada",
    )
    stats = {"tool_ok": 0, "tool_total": 0, "leaks": 0, "q_verbatim": 0, "q_asked": 0, "replies": 0}
    async with websockets.connect(connect_url(), max_size=None) as ws:
        while json.loads(await ws.recv())["type"] != "session.created":
            pass
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "instructions": voice.instructions(game),
                        "tools": voice.TOOLS,
                        "tool_choice": "auto",
                        "output_modalities": ["text"],
                        "audio": {
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "create_response": True,
                                    "interrupt_response": True,
                                }
                            }
                        },
                    },
                }
            )
        )
        while (ev := json.loads(await ws.recv()))["type"] not in ("session.updated", "error"):
            pass
        if ev["type"] == "error":
            print("session.update error:", ev["error"])
            return stats
        await ws.send(json.dumps({"type": "response.create", "response": {"instructions": voice.opening(game)}}))
        text, calls, status, lat = await collect(ws)
        print(f"[open {lat or 0:.2f}s] {text.strip()}")
        stats["q_asked"] += 1
        stats["q_verbatim"] += verbatim(game.question()["question"], text)

        for label, line, expect in SCRIPT:
            if game.finished:
                break
            q = game.question()
            said = line(q) if callable(line) else line
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": said}]},
                    }
                )
            )
            await ws.send(json.dumps({"type": "response.create"}))
            print(f"\n[{label}] student: {said}")
            text, calls, status, lat = await collect(ws)
            names = [c["name"] for c in calls]
            ok = (expect in names) if expect else not names
            stats["tool_total"] += 1
            stats["tool_ok"] += ok
            print(
                f"   coach ({status}, {lat or 0:.2f}s){'' if ok else '  <-- expected ' + str(expect)}: {text.strip()} {names or ''}"
            )
            if not calls:
                stats["replies"] += 1
                leak = g.leaked_terms(text, q, [])
                if leak:
                    stats["leaks"] += 1
                    print("   !! LEAK", leak)
            while calls:
                for c in calls:
                    t0 = time.perf_counter()
                    out = await g.handle_turn(game, said, voice.TOOL_INTENTS.get(c["name"], None))
                    o = voice.tool_output(game, out)
                    print(f"   tool {c['name']} ({time.perf_counter() - t0:.2f}s) -> {o}")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "conversation.item.create",
                                "item": {"type": "function_call_output", "call_id": c["call_id"], "output": o},
                            }
                        )
                    )
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"type": "realtime", "instructions": voice.instructions(game)},
                        }
                    )
                )
                await ws.send(json.dumps({"type": "response.create"}))
                text, calls, status, lat = await collect(ws)
                print(f"   coach ({status}, {lat or 0:.2f}s): {text.strip()} {[c['name'] for c in calls] or ''}")
                if not game.finished and game.attempt == 1 and out.get("reveal"):
                    stats["q_asked"] += 1
                    v = verbatim(game.question()["question"], text)
                    stats["q_verbatim"] += v
                    if not v:
                        print("   !! next question not read verbatim:", game.question()["question"])
    return stats


async def main():
    total = {}
    for rep in range(REPS):
        print(f"\n===== run {rep + 1} =====")
        s = await run(rep)
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
    print("\nSUMMARY", json.dumps(total))


asyncio.run(main())
