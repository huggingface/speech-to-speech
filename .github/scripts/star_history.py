#!/usr/bin/env python3
"""Fetch stargazer history for a repo and render it as a static SVG.

Runs in CI with the automatic GITHUB_TOKEN. No third-party dependencies.

Usage:
    GITHUB_TOKEN=... python star_history.py owner/repo output.svg
"""

import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

API = "https://api.github.com/graphql"
PER_PAGE = 100
MAX_PAGES = 400  # preserve the REST implementation's 40k-star cap
MAX_POINTS = 120  # downsample the curve to at most this many points


def gh_post(query: str, variables: dict, token: str) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(API, data=payload)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if result.get("errors"):
        raise RuntimeError(json.dumps(result["errors"]))
    return result


def fetch_star_dates(repo: str, token: str) -> list[datetime]:
    owner, name = repo.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $cursor: String, $perPage: Int!) {
      repository(owner: $owner, name: $name) {
        stargazers(first: $perPage, after: $cursor) {
          edges { starredAt }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    dates = []
    page = 1
    cursor = None
    while True:
        result = gh_post(
            query,
            {
                "owner": owner,
                "name": name,
                "cursor": cursor,
                "perPage": PER_PAGE,
            },
            token,
        )
        stargazers = result["data"]["repository"]["stargazers"]
        for edge in stargazers["edges"]:
            ts = edge.get("starredAt")
            if ts:
                dates.append(
                    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                        tzinfo=timezone.utc
                    )
                )
        page_info = stargazers["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        page += 1
        if page > MAX_PAGES:
            break
        cursor = page_info["endCursor"]
    dates.sort()
    return dates


def downsample(dates: list[datetime]) -> list[tuple[datetime, int]]:
    n = len(dates)
    points = [(dates[0], 1)]
    if n > 1:
        step = max(1, n // MAX_POINTS)
        for i in range(step, n, step):
            points.append((dates[i], i + 1))
        points.append((dates[-1], n))
    now = datetime.now(timezone.utc)
    points.append((now, n))
    return points


def nice_ceil(v: float) -> int:
    if v <= 0:
        return 1
    mag = 10 ** int(math.floor(math.log10(v)))
    for mult in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= mag * mult:
            return int(mag * mult)
    return int(mag * 10)


def render_svg(points: list[tuple[datetime, int]], repo: str) -> str:
    width, height = 800, 420
    ml, mr, mt, mb = 70, 30, 50, 60
    pw, ph = width - ml - mr, height - mt - mb

    t0 = points[0][0].timestamp()
    t1 = points[-1][0].timestamp()
    tspan = max(t1 - t0, 1)
    ymax = nice_ceil(points[-1][1] * 1.08)

    def x(ts: float) -> float:
        return ml + (ts - t0) / tspan * pw

    def y(v: float) -> float:
        return mt + ph - v / ymax * ph

    line = " ".join(
        f"{'M' if i == 0 else 'L'}{x(d.timestamp()):.1f},{y(c):.1f}"
        for i, (d, c) in enumerate(points)
    )
    area = (
        line
        + f" L{x(points[-1][0].timestamp()):.1f},{y(0):.1f}"
        + f" L{x(points[0][0].timestamp()):.1f},{y(0):.1f} Z"
    )

    grid, ylabels = [], []
    for i in range(6):
        v = ymax * i / 5
        yy = y(v)
        grid.append(
            f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
            f'stroke="#8b949e" stroke-opacity="0.25" stroke-width="1"/>'
        )
        label = f"{v / 1000:.1f}k".replace(".0k", "k") if v >= 1000 else f"{int(v)}"
        ylabels.append(
            f'<text x="{ml - 10}" y="{yy + 4:.1f}" text-anchor="end" '
            f'class="lbl">{label}</text>'
        )

    xlabels = []
    for i in range(6):
        ts = t0 + tspan * i / 5
        d = datetime.fromtimestamp(ts, tz=timezone.utc)
        xlabels.append(
            f'<text x="{x(ts):.1f}" y="{mt + ph + 22}" text-anchor="middle" '
            f'class="lbl">{d.strftime("%b %Y")}</text>'
        )

    total = points[-1][1]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="-apple-system,'Segoe UI',Helvetica,Arial,sans-serif">
  <style>
    .lbl {{ font-size: 12px; fill: #8b949e; }}
    .title {{ font-size: 16px; font-weight: 600; fill: #8b949e; }}
    .total {{ font-size: 13px; fill: #8b949e; }}
  </style>
  <text x="{ml}" y="28" class="title">{repo} star history</text>
  <text x="{ml + pw}" y="28" text-anchor="end" class="total">{total:,} stars</text>
  {"".join(grid)}
  {"".join(ylabels)}
  {"".join(xlabels)}
  <path d="{area}" fill="#f4b400" fill-opacity="0.12"/>
  <path d="{line}" fill="none" stroke="#f4b400" stroke-width="2.5" stroke-linejoin="round"/>
  <circle cx="{x(points[-1][0].timestamp()):.1f}" cy="{y(total):.1f}" r="4" fill="#f4b400"/>
</svg>
"""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo, out = sys.argv[1], sys.argv[2]
    token = os.environ.get("GITHUB_TOKEN", "")
    dates = fetch_star_dates(repo, token)
    if not dates:
        print("no stargazers found", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    svg = render_svg(downsample(dates), repo)
    with open(out, "w") as f:
        f.write(svg)
    print(f"wrote {out} ({len(dates):,} stars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
