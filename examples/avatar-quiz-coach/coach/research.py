"""Turn a topic into a quiz pack: gather sources, then have an LLM write questions.

A topic that links to (or names) a codebase or a paper is read first-hand
(see sources.py). Other topics use Wikipedia (always, no key needed) and, when
SERPER_API_KEY is set, Google results whose pages are fetched and reduced to
plain text. The writer model lists the topic's key concepts, then writes a
few more spoken-friendly questions than needed (reference answer, key points,
hint, fun fact); near-duplicates are dropped before the quiz is assembled.
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

from . import llm, sources

logger = logging.getLogger(__name__)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "").strip()
USER_AGENT = "VivaQuizCoach/0.1 (speech-to-speech example; https://github.com/huggingface/speech-to-speech)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_SOURCE_CHARS = 7000
MAX_TOTAL_CHARS = 26000
# A repository or paper read first-hand gets a larger reading budget.
MAX_PRIMARY_CHARS = 80000
# Bumped when research changes, so cached packs from older research are redone.
RESEARCH_VERSION = 5
SPARE_QUESTIONS = 3

DIFFICULTIES = {
    "rookie": "Beginner: the core ideas and what things are for, in plain words. Mostly difficulty 1, a few 2.",
    "pro": (
        "Intermediate: how and why things work, how the parts interact, comparisons. "
        "At most one pure definition question. Mostly difficulty 2, some 1 and 3."
    ),
    "expert": (
        "Expert: trade-offs, failure modes, subtle interactions between parts, and what would change "
        "if a key design choice were different. Mostly difficulty 3, some 2."
    ),
}

KIND_GUIDES = {
    "codebase": (
        "The material is a software project: its README, docs, a map of its layout and outlines of its main "
        "source files. The student is a developer who wants to understand this codebase. Test what the project "
        "is for and how it is built: how data or a request flows through its components, the key abstractions "
        "and why they exist, important design decisions and their trade-offs (latency, quality, memory, "
        "extensibility), and how its main features work. Name components by plain spoken descriptions, not code "
        "identifiers, and never ask for flag spellings, file paths, default values, version numbers or signatures."
    ),
    "paper": (
        "The material is a research paper. Test the problem and why earlier approaches fall short, the core idea "
        "and how the method works, why its key design choices were made, what the main results show, its "
        "limitations, and how it relates to the work it builds on. Avoid exact table numbers, hyperparameters, "
        "author names and dataset trivia unless they are a headline result."
    ),
}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "topic"


def pack_id_for(topic: str, difficulty: str, count: int) -> str:
    key = f"{' '.join(topic.lower().split())}|{difficulty}|{count}|v{RESEARCH_VERSION}"
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
    "Mark as required only what the question's own wording asks for, with the fewest key points that cover it: one "
    'key point per thing asked. "What is X?", "Where does X come from?" and "Why...?" each ask one thing; "What is '
    'X, and why...?" asks two; "What two..." asks two. A point that explains, justifies or adds detail to another '
    "point is never required, however true or important. Details, examples, names, numbers and consequences the "
    "question does not ask about are never required. When several answers are equally valid (any one reason, cause "
    'or example), write them as one key point joined by "or" and require that point, never one particular '
    'alternative. Write key points as plain facts, without words like "reason" or "example". Every question has at '
    "least one required key point."
)

WRITER_SYSTEM = (
    """You are an expert examiner preparing a SPOKEN quiz. A voice avatar will read each question aloud and a student will answer out loud, so:
- Ask open questions answerable in one or two spoken sentences. Never multiple choice, never "which of the following", no true/false.
- Write every answer, key point, hint, and fun fact in plain, everyday spoken English (like Simplified Technical English / ASD-STE100, but warm and natural, the way a sharp friend explains an idea aloud). Use short, active sentences (10 to 18 words, one idea per sentence) and simple common words. Never sound like a textbook, encyclopedia, or AI assistant: avoid stiff words like "mitigate", "facilitate", "utilize", "exhibit", "leverage", "fundamental", "crucial", "enabling", "thereby", or stacked multi-clause sentences. Keep technical terms only when the question is about them, and explain them plainly.
- No formulas, code, symbols, tables or long lists in questions or answers. Write numbers and acronyms the way people say them.
- Each question tests one clear idea, and every question tests a DIFFERENT idea: no two questions share an answer, and none rephrases or is a sub-part of another.
- Ask about what matters most in the material, not incidental details. No trick questions and no trivia (exact dates, counts, defaults, names) unless it is the point.
- Base every fact on the provided sources; prefer what multiple sources agree on.
- First list the concepts to test, most important first, one per question; then write the questions in that order.
- """
    + REQUIRED_RULE
    + """

Return ONLY a JSON object:
{
  "title": "short display title for the topic (max 40 chars)",
  "emoji": "one emoji for the topic",
  "brief": "one or two short plain-English sentences the coach says before starting: what this topic is and why it matters",
  "concepts": ["the ideas to test, most important first"],
  "questions": [
    {
      "concept": "the concept this question tests",
      "question": "the question, max 30 words",
      "answer": "plain spoken model answer: one or two short, active sentences in everyday words",
      "key_points": ["2 to 4 short atomic facts in plain everyday words"],
      "required": [indices of the key points the question explicitly asks for],
      "hint": "a short plain nudge that does not give the answer away",
      "fun_fact": "one short surprising related fact in plain speech",
      "difficulty": 1,
      "source": 1
    }
  ]
}"""
)


PLANNER_SYSTEM = """A student wants to be quizzed on a topic. Plan the research.
Return ONLY JSON: {"kind": "paper" if the topic is one specific research paper, "codebase" if it is one specific software project or repository, otherwise "general", "paper": "the paper's exact title when kind is paper, else empty", "repo": "owner/name of the project's GitHub repository when kind is codebase and you know it, else empty", "wikipedia": ["1 to 3 exact English Wikipedia article titles that best cover the topic, most relevant first"], "web": "one web search query", "focus": "one sentence: what the quiz should concentrate on, keeping any specifics the student asked for"}"""


async def plan(topic: str) -> dict[str, Any]:
    fallback = {"kind": "general", "paper": "", "repo": "", "wikipedia": [topic], "web": topic, "focus": ""}
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
    kind = str(raw.get("kind") or "general").strip().lower()
    return {
        "kind": kind if kind in ("paper", "codebase") else "general",
        "paper": str(raw.get("paper") or "").strip(),
        "repo": str(raw.get("repo") or "").strip().removeprefix("https://github.com/").strip("/"),
        "wikipedia": wiki or [topic],
        "web": str(raw.get("web") or topic).strip(),
        "focus": str(raw.get("focus") or "").strip(),
    }


def _writer_prompt(
    topic: str,
    difficulty: str,
    count: int,
    sources: list[dict[str, Any]],
    focus: str = "",
    kind: str = "general",
) -> str:
    budget = MAX_PRIMARY_CHARS if kind in KIND_GUIDES else MAX_TOTAL_CHARS
    blocks = []
    for i, src in enumerate(sources, 1):
        text = src.get("text", "")[: max(800, budget)]
        budget -= len(text)
        blocks.append(f"[{i}] {src['title']} ({src['url']})\n{text}")
        if budget <= 0:
            break
    return (
        f"Topic: {topic}\n"
        + (f"Focus: {focus}\n" if focus else "")
        + (f"Material: {KIND_GUIDES[kind]}\n" if kind in KIND_GUIDES else "")
        + f"Level: {DIFFICULTIES.get(difficulty, DIFFICULTIES['pro'])}\n"
        f"Write exactly {count + SPARE_QUESTIONS} questions (the best {count} will be used).\n\n"
        "Sources:\n\n" + "\n\n".join(blocks)
    )


_STOP = set(
    "what which when where does that this with from their there they have into about between would could should "
    "each other than then them these those because while after before over such more most make makes made used "
    "uses using role main play plays part your only also just like been being were will very much many some "
    "explain describe name give tell".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOP}


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two questions that test the same thing (same concept, or mostly the same words)."""
    ca, cb = _words(a.get("concept", "")), _words(b.get("concept", ""))
    if ca and ca == cb:
        return True
    qa, qb = _words(a["question"]), _words(b["question"])
    aa, ab = _words(a["answer"]), _words(b["answer"])
    return _overlap(qa, qb) >= 0.7 or (_overlap(aa, ab) >= 0.6 and _overlap(qa, qb) >= 0.4)


def pick_distinct(questions: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """The first ``count`` questions (most important first) that don't repeat an earlier one."""
    chosen: list[dict[str, Any]] = []
    for q in questions:
        if not any(similar(q, c) for c in chosen):
            chosen.append(q)
        if len(chosen) == count:
            break
    return chosen


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
                "concept": str(q.get("concept", "")).strip(),
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
    questions = sorted(pick_distinct(questions, count), key=lambda q: q["difficulty"])
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
    kind = "general"
    found: list[dict[str, Any]] = []
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=12.0) as http:
        targets = sources.find_targets(topic) or await _named_targets(http, research_plan)
        for target in targets[:2]:
            async for event in _read_target(http, target, found):
                yield event
        if found:
            kinds = {s.get("kind") for s in found}
            kind = "codebase" if "codebase" in kinds else "paper" if "paper" in kinds else "general"
        else:
            async for event in _search(http, topic, research_plan, found):
                yield event

    yield {"type": "status", "stage": "write", "message": "Writing your questions"}
    messages = [
        {"role": "system", "content": WRITER_SYSTEM},
        {
            "role": "user",
            "content": _writer_prompt(topic, difficulty, count, found, research_plan.get("focus", ""), kind),
        },
    ]
    raw, model = await llm.chat_json(messages, llm.WRITER_MODELS, max_tokens=10000, temperature=0.5, timeout=150)
    pack = normalize_pack(raw, topic=topic, difficulty=difficulty, count=count)
    pack.update(
        {
            "id": pack_id_for(topic, difficulty, count),
            "topic": topic,
            "slug": slugify(topic),
            "difficulty": difficulty,
            "kind": kind,
            "model": model,
            "sources": [{"title": s["title"], "url": s["url"], "site": s["site"]} for s in found],
        }
    )
    yield {"type": "pack", "pack": pack}


async def _named_targets(http: httpx.AsyncClient, research_plan: dict[str, Any]) -> list[sources.Target]:
    """A paper or repository the topic names without linking it."""
    try:
        if research_plan.get("kind") == "paper" and research_plan.get("paper"):
            arxiv_id = await sources.arxiv_search(http, research_plan["paper"])
            if arxiv_id:
                return [sources.Target("arxiv", f"https://arxiv.org/abs/{arxiv_id}", arxiv_id=arxiv_id)]
        repo = research_plan.get("repo") or ""
        if research_plan.get("kind") == "codebase" and await sources.repo_exists(http, repo):
            owner, name = repo.split("/", 1)
            return [sources.Target("github", f"https://github.com/{repo}", owner, name)]
    except httpx.HTTPError as exc:
        logger.warning("could not resolve a named paper or repo: %s", exc)
    return []


async def _read_target(
    http: httpx.AsyncClient, target: sources.Target, found: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    try:
        if target.kind == "github":
            name = f"{target.owner}/{target.repo}" + (f"/{target.path}" if target.path else "")
            yield {"type": "status", "stage": "read", "message": f"Reading the {name} repository"}
            got, _ = await sources.github_sources(http, target)
            for s in got:
                s["kind"] = "codebase"
            code = [s for s in got if s.get("code")]
            for s in got:
                if not s.get("code"):
                    yield {"type": "source", "title": s["title"], "url": s["url"], "site": s["site"]}
            if code:
                yield {
                    "type": "source",
                    "title": f"{len(code)} source files, outlined",
                    "url": target.url,
                    "site": "github.com",
                }
        elif target.kind == "arxiv":
            yield {"type": "status", "stage": "read", "message": f"Reading arXiv paper {target.arxiv_id}"}
            got, _ = await sources.arxiv_sources(http, target, html_to_text)
            for s in got:
                s["kind"] = "paper"
                yield {"type": "source", "title": s["title"], "url": s["url"], "site": s["site"]}
        else:
            yield {"type": "status", "stage": "read", "message": f"Reading {target.url}"}
            got = await sources.page_source(http, target, html_to_text)
            for s in got:
                yield {"type": "source", "title": s["title"], "url": s["url"], "site": s["site"]}
        found.extend(got)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("could not read %s: %s", target.url, exc)
        yield {"type": "status", "stage": "read", "message": f"Couldn't read {target.url}"}


async def _search(
    http: httpx.AsyncClient, topic: str, research_plan: dict[str, Any], found: list[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    yield {"type": "status", "stage": "search", "message": f"Searching for “{topic}”"}
    tasks = [_wikipedia(http, research_plan["wikipedia"])]
    if SERPER_API_KEY:
        tasks.append(_serper(http, research_plan["web"]))
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
        found[:] = [by_url.get(s["url"], s) for s in found]


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
