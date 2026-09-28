"""Suggest GitHub logins for Reddit usernames already stored in a run.

Usage:
  python3 suggest_identity.py --run r2008-v2 --top 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.limits import TIMEOUT, USER_AGENT
from osi.store import get_run, list_runs, load_metrics

# Unauthenticated GitHub allows 60 requests an hour. 1.5s keeps a short run under that budget.
_PAUSE_SECONDS = 1.5
# A profile hit is enough. Stop well before a long scan can spend the whole hour.
_MATCH_LIMIT = 100
_OUTPUT = ROOT / "identity_candidates.json"


class _RateLimited(Exception):
    """GitHub answered 403 or 429. The scan must stop."""


def _auto_generated(name: str) -> bool:
    """Digits, bots, and throwaways are not stable public handles."""
    lowered = name.lower()
    if any(character.isdigit() for character in name):
        return True
    return "bot" in lowered or "throwaway" in lowered


def _login_forms(name: str) -> list[str]:
    """Original casing, then lowercase when they differ.

    A 200 from the first form already includes GitHub's canonical login, so the
    second URL is only requested when that casing is missing.
    """
    lowered = name.lower()
    if name == lowered:
        return [name]
    return [name, lowered]


def _fetch_profile(login: str) -> dict | None:
    """GET /users/{login}. None means the account is not there. Followers come from this JSON."""
    # Quote the path segment so a slash in a Reddit name cannot change the URL.
    url = "https://api.github.com/users/" + urllib.parse.quote(login, safe="")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        # 403 is GitHub's unauthenticated rate-limit response. 429 is the other stop signal.
        if error.code in (403, 429):
            raise _RateLimited from None
        detail = error.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"upstream HTTP {error.code} for {url}: {detail}") from None


def _ranked_names(run_id: str) -> list[tuple[str, float]] | None:
    meta = get_run(run_id)
    if meta is None:
        for saved_id, source, _created_at, _notes in list_runs():
            print(f"{saved_id} {source}")
        return None
    scores = load_metrics(run_id, "pagerank")
    # load_metrics is already strongest-first. The list is the Reddit node ids.
    return [(str(node), float(score)) for node, score in scores.items()]


def _print_table(rows: list[dict]) -> None:
    headers = ("reddit_handle", "github_handle", "gh_followers", "reddit_pagerank")
    rendered = []
    for row in rows:
        rendered.append(
            (
                row["reddit"],
                row["github"],
                str(row["followers"]),
                f"{row['pagerank']:.6f}",
            )
        )
    widths = [len(header) for header in headers]
    for cells in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    for cells in rendered:
        print("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))


def _write_candidates(rows: list[dict]) -> None:
    # Same shape as identity.json. The person id is the Reddit username.
    payload = {row["reddit"]: {"github": row["github"], "reddit": row["reddit"]} for row in rows}
    _OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def suggest(run_id: str, top: int) -> int:
    ranked = _ranked_names(run_id)
    if ranked is None:
        return 1
    matches: list[dict] = []
    checked = 0
    requests = 0
    limited = False
    for name, score in ranked[:top]:
        if _auto_generated(name):
            continue
        if len(matches) >= _MATCH_LIMIT:
            break
        checked += 1
        profile = None
        for login in _login_forms(name):
            # Pause before every request except the first, including the lowercase retry.
            if requests:
                time.sleep(_PAUSE_SECONDS)
            requests += 1
            try:
                profile = _fetch_profile(login)
            except _RateLimited:
                limited = True
                break
            if profile is not None:
                break
        if limited:
            break
        if not profile or not profile.get("login"):
            continue
        matches.append(
            {
                "reddit": name,
                "github": str(profile["login"]),
                "name": profile.get("name"),
                "followers": int(profile.get("followers") or 0),
                "pagerank": score,
            }
        )
    # Keep the PageRank order even if a later filter is added.
    matches.sort(key=lambda row: (-row["pagerank"], row["reddit"]))
    _print_table(matches)
    _write_candidates(matches)
    if limited:
        print(f"rate limited after checking {checked}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest GitHub handles for Reddit usernames in a saved run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be at least 1")
    return suggest(args.run, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
