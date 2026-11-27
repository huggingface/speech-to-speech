"""Bake Reachy Mini's recorded emotions into the small clip file the browser plays.

Downloads the moves the coach uses from Pollen Robotics' emotions library
(https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library,
Apache-2.0) and writes static/avatars/reachy-moves.json: per move, 25 fps frames
of [x y z qx qy qz qw antR antL yaw] (head pose in the robot's base frame, in
metres, as the SDK sends it; antennas and body yaw in radians). Each move is
turned about the vertical so that its body starts facing forward, so the
robot keeps facing the student. Each move's sound (the .ogg the robot plays
with it) goes to static/avatars/reachy-sounds/<name>.ogg.

    python scripts/bake_reachy_moves.py
"""

import json
import math
import sys
import urllib.request
from pathlib import Path

DATASET = "https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library/resolve/main"
OUT = Path(__file__).resolve().parent.parent / "static" / "avatars" / "reachy-moves.json"
SOUNDS = OUT.parent / "reachy-sounds"
FPS = 25

# A name, or (alias, name, start s, end s) for a slice: "sleep" is the
# resting pose that wake-mini-up starts from.
MOVES = [
    ("sleep", "wake-mini-up", 0.0, 0.0), "welcoming1", "welcoming2",
    "success1", "success2", "proud2", "yes1", "cheerful1", "enthusiastic1", "enthusiastic2", "dance1", "laughing1",
    "inquiring1", "understanding1", "uncertain1", "oops2", "helpful1", "resigned1",
    "thoughtful1", "thoughtful2", "attentive1", "attentive2", "amazed1",
]


def quat(m):
    """Rotation matrix (3x3, row-major) to quaternion (x, y, z, w)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s)


def frame(target, yaw0):
    h = target["head"]
    # Turn the whole pose (head and body) by -yaw0 about z.
    c, s = math.cos(-yaw0), math.sin(-yaw0)
    rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    r = [[sum(rz[i][k] * h[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    p = [sum(rz[i][k] * h[k][3] for k in range(3)) for i in range(3)]
    q = quat(r)
    if q[3] < 0:
        q = tuple(-c for c in q)
    ant = target.get("antennas") or [0.0, 0.0]
    vals = [*p, *q, ant[0], ant[1], target.get("body_yaw", 0.0) - yaw0]
    return [round(v, 4) for v in vals]


def bake(spec):
    alias, name, t0, t1 = spec if isinstance(spec, tuple) else (spec, spec, 0.0, None)
    with urllib.request.urlopen(f"{DATASET}/{name}.json", timeout=60) as r:
        data = json.load(r)
    times, targets = data["time"], data["set_target_data"]
    yaw0 = targets[0].get("body_yaw", 0.0)
    end = times[-1] if t1 is None else t1
    frames, i = [], 0
    for k in range(int((end - t0) * FPS) + 1):
        t = t0 + k / FPS
        while i + 1 < len(times) and times[i + 1] <= t:
            i += 1
        frames.append(frame(targets[i], yaw0))
    return {"name": alias, "fps": FPS, "description": data.get("description", ""), "frames": frames}


def fetch_sound(spec):
    """The sound that plays with a move (not for slices like "sleep")."""
    if isinstance(spec, tuple):
        return
    SOUNDS.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{DATASET}/{spec}.ogg", timeout=60) as r:
        (SOUNDS / f"{spec}.ogg").write_bytes(r.read())


def main():
    moves = []
    for spec in MOVES:
        moves.append(bake(spec))
        fetch_sound(spec)
        print(f"{moves[-1]['name']}: {len(moves[-1]['frames']) / FPS:.1f} s", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": DATASET.rsplit("/resolve", 1)[0], "license": "Apache-2.0", "moves": moves}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
