"""Mark which key points each featured question actually asks for (``required``).

Key points mix what a question asks with extra context; graders that treat
every key point as required mark brief, correct answers as partial.

    HF_TOKEN=... python scripts/mark_required.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coach import llm  # noqa: E402
from coach.research import REQUIRED_RULE  # noqa: E402

SYSTEM = f"""For each quiz question, decide which of its key points the question itself asks for.
{REQUIRED_RULE}
Return ONLY JSON: {{"required": [[indices for question 0], [indices for question 1], ...]}}"""


async def mark(path: Path) -> None:
    pack = json.loads(path.read_text())
    items = [{"question": q["question"], "key_points": q["key_points"]} for q in pack["questions"]]
    raw, model = await llm.chat_json(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        llm.WRITER_MODELS,
        max_tokens=2000,
        temperature=0.1,
        timeout=90,
    )
    for q, req in zip(pack["questions"], raw.get("required") or []):
        valid = sorted({int(i) for i in req if str(i).isdigit() and int(i) < len(q["key_points"])})
        q["required"] = valid or list(range(len(q["key_points"])))
    path.write_text(json.dumps(pack, ensure_ascii=False, indent=1) + "\n")
    print(
        f"{path.name} via {model}: "
        + " ".join(f"{len(q['required'])}/{len(q['key_points'])}" for q in pack["questions"])
    )


async def main() -> None:
    await asyncio.gather(*(mark(p) for p in sorted((ROOT / "packs").glob("*.json"))))


if __name__ == "__main__":
    asyncio.run(main())
