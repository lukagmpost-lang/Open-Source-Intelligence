"""Public GitHub follow graph. Runnable as a script and importable."""

from __future__ import annotations

import os
import sys
from typing import Any

import networkx as nx

from osi.limits import GITHUB_MAX_PAGES, GITHUB_PAGE_SIZE, get_json


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _page(username: str, relation: str, page: int) -> list[dict[str, Any]]:
    url = (
        f"https://api.github.com/users/{username}/{relation}"
        f"?per_page={GITHUB_PAGE_SIZE}&page={page}"
    )
    payload = get_json(url, _headers())
    if not isinstance(payload, list):
        raise RuntimeError(f"GitHub {relation} for {username} was not public")
    return payload


def _logins(username: str, relation: str) -> set[str]:
    found: set[str] = set()
    for page in range(1, GITHUB_MAX_PAGES + 1):
        batch = _page(username, relation, page)
        for account in batch:
            login = account.get("login")
            if login:
                found.add(login)
        if len(batch) < GITHUB_PAGE_SIZE:
            break
    return found


def fetch_github_graph(username: str) -> nx.Graph:
    """Followers, following, and mutuals for one public GitHub login.

    The result is an undirected ``networkx.Graph``. Each edge has ``relation``
    of ``follow`` or ``mutual``, plus ``direction`` (``follower``, ``following``,
    or ``mutual``) so a caller can recover who follows whom.
    """
    username = username.strip()
    if not username or "/" in username:
        raise ValueError("username must be a single public GitHub login")
    profile = get_json(f"https://api.github.com/users/{username}", _headers())
    if profile.get("message") and "login" not in profile:
        raise RuntimeError(profile["message"])
    login = profile.get("login", username)
    followers = _logins(login, "followers")
    following = _logins(login, "following")

    graph = nx.Graph()
    graph.add_node(login, platform="github", label=login, public=True, center=True)
    for other in followers | following:
        if other == login:
            continue
        mutual = other in followers and other in following
        if mutual:
            direction = "mutual"
            relation = "mutual"
        elif other in followers:
            direction = "follower"
            relation = "follow"
        else:
            direction = "following"
            relation = "follow"
        graph.add_node(other, platform="github", label=other, public=True, center=False)
        graph.add_edge(login, other, relation=relation, direction=direction, weight=1.0, layer=relation)
    return graph


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m osi.github_graph USERNAME", file=sys.stderr)
        return 2
    graph = fetch_github_graph(args[0])
    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    for left, right, data in graph.edges(data=True):
        print(f"{left} {data['relation']} {right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
