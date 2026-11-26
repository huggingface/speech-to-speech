"""Pre-generate the featured topic packs into ../packs so they start instantly.

HF_TOKEN=... python scripts/bake_packs.py            # all missing packs
HF_TOKEN=... python scripts/bake_packs.py --force voice-ai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coach import research  # noqa: E402

FEATURED = [
    {
        "slug": "transformers",
        "topic": "Transformers and the attention mechanism",
        "tagline": "Queries, keys, values and why attention won",
        "wikipedia": ["Transformer (deep learning)", "Attention (machine learning)", "Attention Is All You Need"],
        "focus": "How transformers and self-attention work and why they replaced recurrent networks.",
    },
    {
        "slug": "llm-training",
        "topic": "How large language models are trained",
        "tagline": "Pretraining, fine-tuning, RLHF and DPO",
        "wikipedia": [
            "Large language model",
            "Reinforcement learning from human feedback",
            "Fine-tuning (deep learning)",
        ],
        "focus": "The training pipeline of LLMs: pretraining on next-token prediction, instruction tuning, and preference tuning with RLHF or DPO.",
    },
    {
        "slug": "voice-ai",
        "topic": "Voice AI: speech recognition, text-to-speech and voice agents",
        "tagline": "The tech that is talking to you right now",
        "wikipedia": ["Speech recognition", "Speech synthesis", "Voice activity detection"],
        "focus": "How speech-to-text, text-to-speech and voice activity detection work, and how they combine into a real-time voice assistant.",
    },
    {
        "slug": "diffusion",
        "topic": "Diffusion models for image generation",
        "tagline": "From noise to pictures",
        "wikipedia": ["Diffusion model", "Stable Diffusion", "Text-to-image model"],
        "focus": "How diffusion models generate images by learning to reverse a noising process, and how text conditioning works.",
    },
    {
        "slug": "ai-agents",
        "topic": "AI agents and tool use",
        "tagline": "When LLMs start taking actions",
        "wikipedia": ["Intelligent agent", "Large language model", "Retrieval-augmented generation"],
        "focus": "How LLM-based agents plan, call tools and functions, use retrieval, and the risks and limits of autonomy.",
    },
    {
        "slug": "open-source-ai",
        "topic": "Open-source AI and Hugging Face",
        "tagline": "Open weights, the Hub and the community",
        "wikipedia": ["Hugging Face", "Open-source artificial intelligence", "Llama (language model)"],
        "focus": "What open-source and open-weight AI means, the role of Hugging Face and its Hub, and notable open models.",
    },
    {
        "slug": "ai-history",
        "topic": "The history of artificial intelligence",
        "tagline": "From Turing to ChatGPT",
        "wikipedia": ["History of artificial intelligence", "Turing test", "AI winter"],
        "focus": "Milestones of AI history: Turing, the Dartmouth workshop, AI winters, deep learning's rise and the arrival of chatbots.",
    },
    {
        "slug": "french-revolution",
        "topic": "The French Revolution",
        "tagline": "Liberty, equality, and a lot of drama",
        "wikipedia": ["French Revolution", "Reign of Terror", "Storming of the Bastille"],
        "focus": "Causes, key events and figures, the Terror, and the legacy of the French Revolution.",
    },
]


async def bake(entry: dict, order: int, force: bool) -> None:
    out = ROOT / "packs" / f"{entry['slug']}.json"
    if out.exists() and not force:
        print(f"skip {entry['slug']} (exists)")
        return
    plan = {"wikipedia": entry["wikipedia"], "web": entry["topic"], "focus": entry["focus"]}
    async for event in research.build_pack(entry["topic"], "pro", 8, research_plan=plan):
        if event["type"] != "pack":
            print(f"  {entry['slug']}: {event.get('message') or event.get('title')}")
            continue
        pack = event["pack"]
        pack.update({"slug": entry["slug"], "tagline": entry["tagline"], "order": order})
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(pack, ensure_ascii=False, indent=1) + "\n")
        print(f"wrote {out.name}: {len(pack['questions'])} questions via {pack['model']}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    entries = [(i, e) for i, e in enumerate(FEATURED) if not args.slugs or e["slug"] in args.slugs]
    await asyncio.gather(*(bake(e, i, args.force) for i, e in entries))


if __name__ == "__main__":
    asyncio.run(main())
