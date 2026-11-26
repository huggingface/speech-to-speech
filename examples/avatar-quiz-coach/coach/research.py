"""Turn a topic into a quiz pack: gather sources, then have an LLM write questions.

Sources come from Wikipedia (always, no key needed) and, when SERPER_API_KEY is
set, Google results whose pages are fetched and reduced to plain text. The
writer model sees numbered excerpts and returns spoken-friendly questions with
reference answers, key points, a hint and a fun fact each.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from . import llm

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
USER_AGENT = "VivaQuizCoach/0.1 (speech-to-speech example; https://github.com/huggingface/speech-to-speech)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_SOURCE_CHARS = 7000
MAX_TOTAL_CHARS = 26000

DIFFICULTIES = {
    "rookie": "Beginner level: core definitions and well-known facts. Mostly difficulty 1, a few 2.",
    "pro": "Intermediate level: mechanisms, comparisons and why things work. Mix of difficulty 1-3, mostly 2.",
    "expert": "Expert level: subtle details, trade-offs and precise facts. Mostly difficulty 2 and 3.",
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "topic"


def pack_id_for(topic: str, difficulty: str, count: int) -> str:
    key = f"{' '.join(topic.lower().split())}|{difficulty}|{count}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ── Sources ──────────────────────────────────────────────────────────────────


async def _wikipedia(http: httpx.AsyncClient, queries: list[str], per_query: int = 1) -> list[dict[str, str]]:
    titles: list[str] = []
    for query in queries:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": str(per_query), "format": "json"}
        resp = await http.get(WIKI_API, params=params)
        resp.raise_for_status()
        for hit in resp.json().get("query", {}).get("search", []):
            if hit["title"] not in titles:
                titles.append(hit["title"])
    out: list[dict[str, str]] = []
    for title in titles:
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "redirects": "1",
            "titles": title,
            "format": "json",
        }
        resp = await http.get(WIKI_API, params=params)
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            text = _trim_wiki(page.get("extract") or "")
            if len(text) > 400:
                url = "https://en.wikipedia.org/wiki/" + page.get("title", title).replace(" ", "_")
                out.append({"title": page.get("title", title), "url": url, "site": "wikipedia.org", "text": text})
    return out


def _trim_wiki(text: str) -> str:
    # Drop reference-like tail sections; they carry no quizzable content.
    cut = re.search(r"\n(See also|References|External links|Further reading|Notes)\n", text)
    if cut:
        text = text[: cut.start()]
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_SOURCE_CHARS]


async def _serper(http: httpx.AsyncClient, topic: str, limit: int = 5) -> list[dict[str, str]]:
    resp = await http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": topic, "num": limit + 3},
    )
    resp.raise_for_status()
    results = []
    for item in resp.json().get("organic", []):
        url = item.get("link", "")
        host = urlparse(url).netloc.removeprefix("www.")
        if not url or "wikipedia.org" in host or host.endswith(("youtube.com", "tiktok.com", "instagram.com")):
            continue
        results.append({"title": item.get("title", host), "url": url, "site": host, "text": item.get("snippet", "")})
        if len(results) >= limit:
            break
    return results


_TAG = re.compile(r"<(script|style|noscript|svg|nav|footer|header|form)[^>]*>.*?</\1>", re.S | re.I)


def html_to_text(raw: str) -> str:
    try:
        import trafilatura  # type: ignore[import-not-found]

        text = trafilatura.extract(raw, include_comments=False, include_tables=False)
        if text:
            return text
    except Exception:  # trafilatura is optional; fall back to a crude strip
        pass
    raw = _TAG.sub(" ", raw)
    raw = re.sub(r"<br\s*/?>|</p>|</h\d>|</li>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) > 40)


async def _fetch_page(http: httpx.AsyncClient, source: dict[str, str]) -> dict[str, str]:
    try:
        resp = await http.get(source["url"], follow_redirects=True, timeout=8.0)
        if resp.status_code == 200 and "html" in resp.headers.get("content-type", ""):
            text = await asyncio.to_thread(html_to_text, resp.text[:1_500_000])
            if len(text) > len(source.get("text", "")):
                return {**source, "text": text[:MAX_SOURCE_CHARS]}
    except httpx.HTTPError as exc:
        logger.info("skip %s: %s", source["url"], exc)
    return source


# ── Question writing ─────────────────────────────────────────────────────────

REQUIRED_RULE = (
    "A key point is required only if the question explicitly asks for it: a correct spoken answer must contain it. "
    "Context the question does not ask about (who wrote it, where, extra facts or numbers) is not required. "
    "Every question has at least one required key point."
)

WRITER_SYSTEM = (
    """You are an expert examiner preparing a SPOKEN quiz. A voice avatar will read each question aloud and a student will answer out loud, so:
- Ask open questions answerable in one or two spoken sentences. Never multiple choice, never "which of the following", no true/false.
- No formulas, code, symbols, tables or long lists in questions or answers. Write numbers and acronyms the way people say them.
- Each question tests one clear idea. Avoid trick questions and trivia about exact dates unless the date is the point.
- Base every fact on the provided sources; prefer what multiple sources agree on.
- Order questions from easiest to hardest.
- """
    + REQUIRED_RULE
    + """

Return ONLY a JSON object:
{
  "title": "short display title for the topic (max 40 chars)",
  "emoji": "one emoji for the topic",
  "brief": "two sentences the coach says before starting: what this topic is and why it matters",
  "questions": [
    {
      "question": "the question, max 30 words",
      "answer": "the model answer, one or two sentences",
      "key_points": ["2 to 4 short atomic facts a complete answer contains"],
      "required": [indices of the key points the question explicitly asks for],
      "hint": "a nudge that does not give the answer away",
      "fun_fact": "one short surprising related fact",
      "difficulty": 1,
      "source": 1
    }
  ]
}"""
)


PLANNER_SYSTEM = """A student wants to be quizzed on a topic. Plan the research.
Return ONLY JSON: {"wikipedia": ["1 to 3 exact English Wikipedia article titles that best cover the topic, most relevant first"], "web": "one web search query", "focus": "one sentence: what the quiz should concentrate on, keeping any specifics the student asked for"}"""


async def plan(topic: str) -> dict[str, Any]:
    fallback = {"wikipedia": [topic], "web": topic, "focus": ""}
    try:
        raw, _ = await llm.chat_json(
            [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": topic}],
            llm.GRADER_MODELS,
            max_tokens=600,
            temperature=0.2,
            timeout=12,
        )
    except llm.LLMError as exc:
        logger.warning("planner fallback: %s", exc)
        return fallback
    wiki = [str(t).strip() for t in raw.get("wikipedia") or [] if str(t).strip()][:3]
    return {
        "wikipedia": wiki or [topic],
        "web": str(raw.get("web") or topic).strip(),
        "focus": str(raw.get("focus") or "").strip(),
    }


def _writer_prompt(topic: str, difficulty: str, count: int, sources: list[dict[str, str]], focus: str = "") -> str:
    budget = MAX_TOTAL_CHARS
    blocks = []
    for i, src in enumerate(sources, 1):
        text = src.get("text", "")[: max(800, min(MAX_SOURCE_CHARS, budget))]
        budget -= len(text)
        blocks.append(f"[{i}] {src['title']} ({src['url']})\n{text}")
        if budget <= 0:
            break
    return (
        f"Topic: {topic}\n"
        + (f"Focus: {focus}\n" if focus else "")
        + f"Level: {DIFFICULTIES.get(difficulty, DIFFICULTIES['pro'])}\n"
        f"Write exactly {count} questions.\n\n"
        "Sources:\n\n" + "\n\n".join(blocks)
    )


def normalize_pack(raw: dict[str, Any], *, topic: str, difficulty: str, count: int) -> dict[str, Any]:
    questions = []
    for q in raw.get("questions") or []:
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        answer = str(q.get("answer", "")).strip()
        if not question or not answer:
            continue
        points = [str(p).strip() for p in (q.get("key_points") or []) if str(p).strip()][:4]
        try:
            level = min(3, max(1, int(q.get("difficulty", 2))))
        except (TypeError, ValueError):
            level = 2
        try:
            source = int(q.get("source", 0))
        except (TypeError, ValueError):
            source = 0
        points = points or [answer]
        required = sorted(
            {int(i) for i in (q.get("required") or []) if str(i).isdigit() and int(i) < len(points)}
        ) or list(range(len(points)))
        questions.append(
            {
                "question": question,
                "answer": answer,
                "key_points": points,
                "required": required,
                "hint": str(q.get("hint", "")).strip(),
                "fun_fact": str(q.get("fun_fact", "")).strip(),
                "difficulty": level,
                "source": source,
            }
        )
    if len(questions) < 3:
        raise ValueError(f"writer returned only {len(questions)} usable questions")
    questions = sorted(questions[:count], key=lambda q: q["difficulty"])
    for i, q in enumerate(questions):
        q["id"] = i
    return {
        "title": str(raw.get("title") or topic).strip()[:60],
        "emoji": str(raw.get("emoji") or "🎓").strip()[:8],
        "brief": str(raw.get("brief") or "").strip(),
        "questions": questions,
    }


async def build_pack(
    topic: str, difficulty: str, count: int, research_plan: dict[str, Any] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Yield progress events, ending with ``{"type": "pack", "pack": {...}}``."""
    yield {"type": "status", "stage": "plan", "message": "Planning what to read"}
    research_plan = research_plan or await plan(topic)
    yield {"type": "status", "stage": "search", "message": f"Searching for “{topic}”"}
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=12.0) as http:
        tasks = [_wikipedia(http, research_plan["wikipedia"])]
        if SERPER_API_KEY:
            tasks.append(_serper(http, research_plan["web"]))
        found: list[dict[str, str]] = []
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, BaseException):
                logger.warning("source search failed: %s", result)
                continue
            found.extend(result)
        if not found:
            yield {"type": "status", "stage": "search", "message": "No sources found, using what I already know"}
        for src in found:
            yield {"type": "source", "title": src["title"], "url": src["url"], "site": src["site"]}

        web = [s for s in found if s["site"] != "wikipedia.org"]
        if web:
            yield {"type": "status", "stage": "read", "message": f"Reading {len(found)} sources"}
            fetched = await asyncio.gather(*(_fetch_page(http, s) for s in web[:4]))
            by_url = {s["url"]: s for s in fetched}
            found = [by_url.get(s["url"], s) for s in found]

    yield {"type": "status", "stage": "write", "message": "Writing your questions"}
    messages = [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": _writer_prompt(topic, difficulty, count, found, research_plan.get("focus", ""))},
    ]
    raw, model = await llm.chat_json(messages, llm.WRITER_MODELS, max_tokens=8000, temperature=0.5, timeout=90)
    pack = normalize_pack(raw, topic=topic, difficulty=difficulty, count=count)
    pack.update(
        {
            "id": pack_id_for(topic, difficulty, count),
            "topic": topic,
            "slug": slugify(topic),
            "difficulty": difficulty,
            "model": model,
            "sources": [{"title": s["title"], "url": s["url"], "site": s["site"]} for s in found],
        }
    )
    yield {"type": "pack", "pack": pack}


def load_packs(directory: str) -> dict[str, dict[str, Any]]:
    packs: dict[str, dict[str, Any]] = {}
    if not os.path.isdir(directory):
        return packs
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                pack = json.load(fh)
            packs[pack["id"]] = pack
    return packs
