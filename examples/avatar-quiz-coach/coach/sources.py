"""Primary sources for topics that point at a codebase or a paper.

A topic can carry a link (a GitHub repository or folder, an arXiv or Hugging
Face Papers page, any web page) or name a paper or repository that the planner
resolves. Instead of encyclopedia articles about the general field, the quiz is
then written from the thing itself:

- GitHub: the README, the most useful docs, and a digest of the main source
  files (module and class layout, signatures, docstrings and the longer
  comments, which is where design reasons live), plus a map of the layout.
- Papers: the full text from arXiv's HTML rendering (the PDF when there is
  none), up to the references.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx

logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
URL_RE = re.compile(r"https?://[^\s<>\"'\])]+")
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(?:v\d+)?")

README_CHARS = 14000
DOC_CHARS = 5500
DOCS_MAX = 7
CODE_FILE_CHARS = 2600
CODE_CHARS = 22000
PAPER_CHARS = 48000
PAGE_CHARS = 20000

CODE_EXT = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".mjs",
    ".rs",
    ".go",
    ".java",
    ".kt",
    ".swift",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".rb",
}
DOC_EXT = {".md", ".mdx", ".rst", ".txt"}
SKIP_DIRS = re.compile(
    r"(^|/)(tests?|testing|__tests__|spec|benchmarks?|examples?|demos?|archive|vendor|third_party|node_modules|dist|build|"
    r"migrations|\.github|\.vscode|site-packages|fixtures|assets|static)(/|$)",
    re.I,
)
SKIP_DOCS = re.compile(
    r"(changelog|license|code_of_conduct|contributing|security|agents|claude|releases?/|history)", re.I
)
GENERIC_STEMS = {
    "utils",
    "util",
    "base",
    "main",
    "config",
    "types",
    "constants",
    "common",
    "helpers",
    "index",
    "init",
    "app",
}


@dataclass
class Target:
    kind: str  # "github" | "arxiv" | "web"
    url: str
    owner: str = ""
    repo: str = ""
    ref: str = ""
    path: str = ""
    arxiv_id: str = ""


def find_targets(topic: str) -> list[Target]:
    """Links in the topic that the researcher should read first-hand."""
    out: list[Target] = []
    for raw in URL_RE.findall(topic):
        url = raw.rstrip(".,;:!?")
        u = urlparse(url)
        host = u.netloc.lower().removeprefix("www.")
        parts = [p for p in u.path.split("/") if p]
        if host == "github.com" and len(parts) >= 2:
            owner, repo = parts[0], parts[1].removesuffix(".git")
            ref = path = ""
            if len(parts) >= 4 and parts[2] in ("tree", "blob"):
                ref, path = parts[3], "/".join(parts[4:])
            out.append(Target("github", f"https://github.com/{owner}/{repo}", owner, repo, ref, path))
        elif (
            host.endswith("arxiv.org")
            or (host == "huggingface.co" and parts[:1] == ["papers"])
            or host == "alphaxiv.org"
        ):
            m = ARXIV_ID.search(u.path)
            if m:
                out.append(Target("arxiv", f"https://arxiv.org/abs/{m.group(1)}", arxiv_id=m.group(1)))
        else:
            out.append(Target("web", url))
    return out


def _source(title: str, url: str, site: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"title": title, "url": url, "site": site, "text": text, **extra}


# ── GitHub ───────────────────────────────────────────────────────────────────


def _gh_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def repo_exists(http: httpx.AsyncClient, full_name: str) -> bool:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", full_name):
        return False
    resp = await http.get(f"https://api.github.com/repos/{full_name}", headers=_gh_headers())
    return resp.status_code == 200


def _ext(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _stem(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()


def rank_docs(paths: list[tuple[str, int]], scope: str = "") -> list[str]:
    """Most useful docs first: component READMEs and design docs, not changelogs."""
    scored = []
    for path, size in paths:
        low = path.lower()
        if _ext(path) not in DOC_EXT or SKIP_DOCS.search(low) or SKIP_DIRS.search(low) or size < 400:
            continue
        if low == (scope + "/readme.md").lstrip("/") or (not scope and "/" not in low and low.startswith("readme")):
            continue  # the main README is read separately
        score = 0.0
        if re.search(r"architecture|design|overview|internals|how[-_]it[-_]works|pipeline|latency|concepts", low):
            score += 3
        if low.startswith("docs/") or "/docs/" in low:
            score += 1.5
        if _stem(path) == "readme":
            score += 2  # a component's README
        score -= 0.4 * low.count("/")
        score += min(1.5, math.log10(max(size, 1)) - 3)
        scored.append((score, path))
    return [p for _, p in sorted(scored, key=lambda x: -x[0])]


def rank_code(paths: list[tuple[str, int]], mentions: str, package_dirs: set[str]) -> list[str]:
    """Central source files first: big, inside the main package, named in the docs."""
    scored = []
    for path, size in paths:
        low = path.lower()
        name = low.rsplit("/", 1)[-1]
        if _ext(path) not in CODE_EXT or SKIP_DIRS.search(low) or size < 600 or size > 250_000:
            continue
        if name.startswith(("test_", "conftest", "setup.")) or name.endswith(
            ("_test.py", ".test.ts", ".spec.ts", ".min.js")
        ):
            continue
        stem = _stem(path)
        score = math.log10(size)
        if stem not in GENERIC_STEMS and len(stem) >= 4 and stem in mentions:
            score += 1.5
        if any(low.startswith(d + "/") for d in package_dirs):
            score += 1
        if stem == "__init__":
            score -= 1.5
        score -= 0.15 * low.count("/")
        scored.append((score, path))
    return [p for _, p in sorted(scored, key=lambda x: -x[0])]


def python_outline(text: str, limit: int = CODE_FILE_CHARS) -> str:
    """Module docstring, classes and functions with their signatures, docstrings and longer comments."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return generic_outline(text, limit)
    lines = text.splitlines()
    out: list[str] = []

    def doc(node: Any, indent: str) -> None:
        d = ast.get_docstring(node)
        if d:
            first = d.strip().split("\n\n")[0]
            out.append(indent + '"""' + " ".join(first.split())[:400] + '"""')

    def comments(node: Any, indent: str) -> None:
        start, end = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
        for ln in lines[start : min(end, start + 400)]:
            s = ln.strip()
            if s.startswith("#") and len(s) > 45 and not s.startswith("#!"):
                out.append(indent + s[:200])

    def func(node: Any, indent: str) -> None:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        out.append(f"{indent}{prefix} {node.name}({ast.unparse(node.args)}){ret}")
        doc(node, indent + "    ")
        comments(node, indent + "    ")

    doc(tree, "")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            out.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
            doc(node, "    ")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not item.name.startswith("_") or item.name in ("__init__", "__call__")
                ):
                    func(item, "    ")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            func(node, "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            comments(node, "")
    return _cap("\n".join(out), limit)


_DECL = re.compile(
    r"^\s*(export\s+)?(default\s+)?(pub(\(crate\))?\s+)?(async\s+)?"
    r"(function|class|interface|type|struct|enum|trait|impl|fn|func|def|module|public|protected)\b"
)
_COMMENT = re.compile(r"^\s*(//|/\*|\*|#)")


def generic_outline(text: str, limit: int = CODE_FILE_CHARS) -> str:
    out = []
    for ln in text.splitlines():
        s = ln.rstrip()
        if _DECL.match(s):
            out.append(s.split("{")[0].rstrip()[:160])
        elif _COMMENT.match(s) and len(s.strip()) > 45:
            out.append(s.strip()[:200])
    return _cap("\n".join(out), limit)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[: cut if cut > limit // 2 else limit] + "\n…"


def layout_map(paths: list[str], scope: str = "", depth: int = 2, max_lines: int = 60) -> str:
    """Directories (two levels) with their file counts and a few file names."""
    groups: dict[str, list[str]] = {}
    for p in paths:
        rel = p[len(scope) + 1 :] if scope and p.startswith(scope + "/") else p
        if SKIP_DIRS.search(rel) and not scope:
            continue
        parts = rel.split("/")
        key = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or "."
        groups.setdefault(key, []).append(parts[-1])
    lines = []
    for key in sorted(groups)[:max_lines]:
        names = [n for n in groups[key] if _ext(n) in CODE_EXT | DOC_EXT][:6]
        lines.append(f"{key}/ ({len(groups[key])} files)" + (": " + ", ".join(names) if names else ""))
    return "\n".join(lines)


async def github_sources(http: httpx.AsyncClient, t: Target) -> tuple[list[dict[str, Any]], str]:
    """Sources for a repository (or one folder of it) and a one-line description."""
    base = f"https://api.github.com/repos/{t.owner}/{t.repo}"
    meta_resp = await http.get(base, headers=_gh_headers())
    meta = meta_resp.json() if meta_resp.status_code == 200 else {}
    ref = t.ref or meta.get("default_branch") or "HEAD"
    full = meta.get("full_name") or f"{t.owner}/{t.repo}"
    raw_base = f"https://raw.githubusercontent.com/{full}/{quote(ref)}/"
    blob_base = f"https://github.com/{full}/blob/{ref}/"
    description = str(meta.get("description") or "")

    tree_resp = await http.get(f"{base}/git/trees/{quote(ref)}", params={"recursive": "1"}, headers=_gh_headers())
    entries = tree_resp.json().get("tree", []) if tree_resp.status_code == 200 else []
    scope = t.path.strip("/")
    blobs = [
        (e["path"], int(e.get("size") or 0))
        for e in entries
        if e.get("type") == "blob" and (not scope or e["path"] == scope or e["path"].startswith(scope + "/"))
    ]
    if not blobs:
        # API unavailable (rate limit) or empty: the README alone still beats nothing.
        logger.warning("github tree unavailable for %s (HTTP %s)", full, tree_resp.status_code)
        blobs = [("README.md", 0)]

    sem = asyncio.Semaphore(8)

    async def raw(path: str) -> str:
        async with sem:
            try:
                resp = await http.get(raw_base + quote(path), follow_redirects=True, timeout=15.0)
            except httpx.HTTPError as exc:
                logger.info("skip %s: %s", path, exc)
                return ""
            return resp.text if resp.status_code == 200 else ""

    readme_path = next(
        (p for p, _ in blobs if p.lower() in ((scope + "/readme.md").lstrip("/"), (scope + "/readme.rst").lstrip("/"))),
        next((p for p, _ in blobs if "/" not in p and p.lower().startswith("readme")), ""),
    )
    docs = rank_docs(blobs, scope)[:DOCS_MAX]
    readme, *doc_texts = await asyncio.gather(
        raw(readme_path) if readme_path else asyncio.sleep(0, ""), *map(raw, docs)
    )
    mentions = (readme + " ".join(doc_texts)).lower()
    package_dirs = {p.rsplit("/", 1)[0] for p, _ in blobs if p.endswith("/__init__.py") and p.count("/") <= 3}
    package_dirs |= {
        d for d in ("src", "lib", "pkg", "internal", "cmd") if any(p.startswith(d + "/") for p, _ in blobs)
    }
    code_paths = rank_code(blobs, mentions, package_dirs)[:24]
    code_texts = await asyncio.gather(*map(raw, code_paths))

    out: list[dict[str, Any]] = []
    layout = layout_map([p for p, _ in blobs], scope)
    head = f"Repository {full}" + (f" ({scope})" if scope else "") + (f": {description}" if description else "")
    out.append(
        _source(
            f"{full} README" + (f" ({scope})" if scope else ""),
            t.url + (f"/tree/{ref}/{scope}" if scope else ""),
            "github.com",
            head + "\n\n" + _cap(readme, README_CHARS) + ("\n\nRepository layout:\n" + layout if layout else ""),
        )
    )
    for path, text in zip(docs, doc_texts, strict=True):
        if len(text) > 300:
            out.append(_source(path, blob_base + path, "github.com", _cap(text, DOC_CHARS)))
    budget = CODE_CHARS
    for path, text in zip(code_paths, code_texts, strict=True):
        if not text or budget < 800:
            continue
        outline = python_outline(text) if path.endswith(".py") else generic_outline(text)
        if len(outline) < 200:
            continue
        outline = _cap(outline, min(CODE_FILE_CHARS, budget))
        budget -= len(outline)
        out.append(_source(path, blob_base + path, "github.com", "Outline of the source file:\n" + outline, code=True))
    return out, description


# ── Papers ───────────────────────────────────────────────────────────────────

_REFS = re.compile(r"\n\s*(References|Bibliography|REFERENCES)\s*\n")


def cut_references(text: str) -> str:
    """Drop the bibliography and what follows (appendices rarely make good spoken questions)."""
    for m in _REFS.finditer(text):
        if m.start() > len(text) * 0.3:
            return text[: m.start()]
    return text


async def arxiv_meta(http: httpx.AsyncClient, arxiv_id: str) -> dict[str, str]:
    resp = await http.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id})
    entry = resp.text.split("<entry>", 1)[-1] if resp.status_code == 200 else ""
    title = re.search(r"<title>(.*?)</title>", entry, re.S)
    summary = re.search(r"<summary>(.*?)</summary>", entry, re.S)
    return {
        "title": " ".join(title.group(1).split()) if title else "",
        "abstract": " ".join(summary.group(1).split()) if summary else "",
    }


_LUCENE_STOP = set(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these "
    "they this to was will with".split()
)


async def arxiv_search(http: httpx.AsyncClient, title: str) -> str:
    """arXiv ID of the paper with this title, or "" when no close match."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    # arXiv's search drops stopwords, and an AND with a dropped term matches nothing.
    terms = [w for w in words if len(w) >= 3 and w not in _LUCENE_STOP][:10]
    if not terms:
        return ""
    query = " AND ".join(f"ti:{w}" for w in terms)
    resp = await http.get("https://export.arxiv.org/api/query", params={"search_query": query, "max_results": "5"})
    if resp.status_code != 200:
        return ""
    want = set(words)
    best, best_score = "", 0.0
    for entry in resp.text.split("<entry>")[1:]:
        t = re.search(r"<title>(.*?)</title>", entry, re.S)
        i = re.search(r"<id>https?://arxiv.org/abs/(\d{4}\.\d{4,5})", entry)
        if not t or not i:
            continue
        got = set(re.findall(r"[a-z0-9]+", t.group(1).lower()))
        score = len(want & got) / max(1, len(want | got))
        if score > best_score:
            best, best_score = i.group(1), score
    return best if best_score >= 0.6 else ""


def _pdf_text(data: bytes) -> str:
    try:
        import io

        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages[:30])
    except Exception as exc:  # malformed PDFs are common enough
        logger.info("pdf extraction failed: %s", exc)
        return ""


async def arxiv_sources(http: httpx.AsyncClient, t: Target, html_to_text: Any) -> tuple[list[dict[str, Any]], str]:
    meta = await arxiv_meta(http, t.arxiv_id)
    title = meta["title"] or f"arXiv {t.arxiv_id}"
    text = ""
    resp = await http.get(f"https://arxiv.org/html/{t.arxiv_id}", follow_redirects=True, timeout=20.0)
    if resp.status_code == 200 and "ltx_" in resp.text:
        text = await asyncio.to_thread(html_to_text, resp.text[:3_000_000])
    if len(text) < 3000:
        pdf = await http.get(f"https://arxiv.org/pdf/{t.arxiv_id}", follow_redirects=True, timeout=30.0)
        if pdf.status_code == 200:
            text = await asyncio.to_thread(_pdf_text, pdf.content) or text
    text = cut_references(text) if text else ""
    if len(text) < 1500:
        text = meta["abstract"]  # last resort: the abstract alone
    body = f"Paper: {title}\n\n" + _cap(text, PAPER_CHARS)
    return [_source(title, t.url, "arxiv.org", body)], meta["abstract"][:300]


async def page_source(http: httpx.AsyncClient, t: Target, html_to_text: Any) -> list[dict[str, Any]]:
    try:
        resp = await http.get(t.url, follow_redirects=True, timeout=15.0)
    except httpx.HTTPError as exc:
        logger.info("skip %s: %s", t.url, exc)
        return []
    if resp.status_code != 200:
        return []
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype:
        text = await asyncio.to_thread(_pdf_text, resp.content)
    elif "html" in ctype:
        text = await asyncio.to_thread(html_to_text, resp.text[:3_000_000])
    else:
        text = resp.text
    host = urlparse(t.url).netloc.removeprefix("www.")
    title = re.search(r"<title[^>]*>(.*?)</title>", resp.text[:20000], re.S | re.I) if "html" in ctype else None
    name = " ".join(title.group(1).split())[:80] if title else host
    return [_source(name, t.url, host, _cap(text, PAGE_CHARS))] if len(text) > 300 else []
