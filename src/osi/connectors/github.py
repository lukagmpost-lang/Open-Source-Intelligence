"""Public GitHub followers and following via the REST API."""

from __future__ import annotations

import os
from typing import Any

from osi.graph import Graph
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


def fetch_github(username: str) -> Graph:
    username = username.strip()
    if not username or "/" in username:
        raise ValueError("username must be a single public GitHub login")
    profile = get_json(f"https://api.github.com/users/{username}", _headers())
    if profile.get("type") == "User" and profile.get("message"):
        raise RuntimeError(profile["message"])
    graph = Graph()
    root = graph.add_node("github", profile.get("login", username), profile.get("login", username), {"public": True})
    followers: set[str] = set()
    following: set[str] = set()
    for relation, bucket in (("followers", followers), ("following", following)):
        for page in range(1, GITHUB_MAX_PAGES + 1):
            batch = _page(username, relation, page)
            for account in batch:
                login = account.get("login")
                if not login:
                    continue
                bucket.add(login)
                graph.add_node("github", login, login)
            if len(batch) < GITHUB_PAGE_SIZE:
                break
    for login in followers:
        graph.add_edge(f"github:{login}", root.id, "follow", 1.0)
    for login in following:
        graph.add_edge(root.id, f"github:{login}", "follow", 1.0)
    for login in followers & following:
        graph.add_edge(root.id, f"github:{login}", "mutual", 1.0)
    return graph
