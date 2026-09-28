"""Public GitHub follow graph. Runnable as a script and importable."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

import networkx as nx

from osi.limits import GITHUB_MAX_PAGES, GITHUB_PAGE_SIZE, get_json

# One request every two seconds. Unauthenticated GitHub allows 60 per hour.
_MULTI_PAUSE_SECONDS = 2.0
# The combined layer and each login's piece share this run. A second call reads it.
MULTI_EGO_RUN = "github-multi-ego"
MULTI_EGO_LAYER = "github"
_last_request = 0.0


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _page(
    username: str,
    relation: str,
    page: int,
    fetch: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    url = (
        f"https://api.github.com/users/{username}/{relation}"
        f"?per_page={GITHUB_PAGE_SIZE}&page={page}"
    )
    getter = fetch or get_json
    payload = getter(url, _headers())
    if not isinstance(payload, list):
        raise RuntimeError(f"GitHub {relation} for {username} was not public")
    return payload


def _logins(
    username: str,
    relation: str,
    max_pages: int | None = None,
    fetch: Callable[..., Any] | None = None,
) -> set[str]:
    found: set[str] = set()
    # None keeps the single-login fetch on the module cap. The multi-ego path passes its own.
    limit = GITHUB_MAX_PAGES if max_pages is None else max_pages
    for page in range(1, limit + 1):
        batch = _page(username, relation, page, fetch=fetch)
        for account in batch:
            login = account.get("login")
            if login:
                found.add(login)
        if len(batch) < GITHUB_PAGE_SIZE:
            break
    return found


def _throttled(url: str, headers: dict[str, str] | None = None) -> Any:
    """GET one GitHub URL, waiting so requests stay two seconds apart."""
    global _last_request
    if _last_request:
        delay = _MULTI_PAUSE_SECONDS - (time.monotonic() - _last_request)
        if delay > 0:
            time.sleep(delay)
    payload = get_json(url, headers if headers is not None else _headers())
    _last_request = time.monotonic()
    return payload


def _normalize(handles: list[str]) -> list[str]:
    """Drop blanks and repeated logins. The first spelling of each name wins."""
    chosen: list[str] = []
    seen: set[str] = set()
    for handle in handles:
        text = str(handle).strip()
        if not text or "/" in text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(text)
    return chosen


def _piece_layer(handle: str) -> str:
    return f"ego:{handle.casefold()}"


def _request_config(handles: list[str], max_pages: int) -> dict[str, Any]:
    return {
        "handles": sorted(handle.casefold() for handle in handles),
        "max_pages": max_pages,
        "layer": MULTI_EGO_LAYER,
    }


def _cached_union(config: dict[str, Any]) -> nx.Graph | None:
    """Return the saved combined graph when this exact handle set was already fetched."""
    from osi.store import get_run, load_graph

    meta = get_run(MULTI_EGO_RUN)
    if meta is None:
        return None
    saved = meta.get("config") or {}
    if saved.get("handles") != config["handles"] or saved.get("max_pages") != config["max_pages"]:
        return None
    graph = load_graph(MULTI_EGO_RUN, MULTI_EGO_LAYER)
    if graph is None:
        return None
    # The run row and the graph can be written separately. Both have to name this handle set.
    if graph.graph.get("handles") != config["handles"] or graph.graph.get("max_pages") != config["max_pages"]:
        return None
    return graph


def _cached_piece(handle: str, max_pages: int) -> nx.Graph | None:
    """Return one login's saved neighborhood, or None when the page cap differs."""
    from osi.store import load_graph

    piece = load_graph(MULTI_EGO_RUN, _piece_layer(handle))
    if piece is None:
        return None
    # A deeper page cap has to hit the API again. A shallower one must not reuse the longer list.
    if piece.graph.get("max_pages") != max_pages:
        return None
    return piece


def _store_piece(handle: str, piece: nx.Graph) -> None:
    from osi.store import save_graph

    save_graph(MULTI_EGO_RUN, _piece_layer(handle), piece)


def _tie(other: str, followers: set[str], following: set[str]) -> dict[str, Any]:
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
    return {"relation": relation, "direction": direction, "weight": 1.0, "layer": relation}


def _merge(combined: nx.Graph, piece: nx.Graph) -> None:
    """Add one ego. A login that is a center in any piece stays a center."""
    for node, data in piece.nodes(data=True):
        if node not in combined:
            combined.add_node(node, **data)
        elif data.get("center"):
            combined.nodes[node]["center"] = True
    for left, right, data in piece.edges(data=True):
        if not combined.has_edge(left, right):
            combined.add_edge(left, right, **data)
            continue
        # The other login's list can record the opposite direction of the same follow.
        current = combined.edges[left, right]
        if current.get("direction") != data.get("direction"):
            current["direction"] = "mutual"
            current["relation"] = "mutual"
            current["layer"] = "mutual"


def _fetch_one(handle: str, max_pages: int) -> nx.Graph:
    """Followers and following for one login. Zero followers produce an empty graph."""
    print(f"fetching {handle}", file=sys.stderr)
    profile = _throttled(f"https://api.github.com/users/{handle}", _headers())
    if profile.get("message") and "login" not in profile:
        raise RuntimeError(profile["message"])
    login = profile.get("login", handle)
    graph = nx.Graph()
    graph.graph["max_pages"] = max_pages
    # No follower list means this login adds no GitHub structure. Candidate files stay as they are.
    if profile.get("followers") == 0:
        print(f"note: {login} has zero followers; skipped on the GitHub graph", file=sys.stderr)
        graph.graph["skipped"] = login
        return graph
    followers = _logins(login, "followers", max_pages=max_pages, fetch=_throttled)
    following = _logins(login, "following", max_pages=max_pages, fetch=_throttled)
    graph.add_node(login, platform="github", label=login, public=True, center=True)
    for other in followers | following:
        if other == login:
            continue
        graph.add_node(other, platform="github", label=other, public=True, center=False)
        graph.add_edge(login, other, **_tie(other, followers, following))
    return graph


def fetch_github_multi_ego(handles: list[str], max_pages: int = 2) -> nx.Graph:
    """Followers and following for many logins, as one undirected graph.

    A neighbor of two centers is one node with two edges, so the layer is not a
    collection of stars. A login with zero followers is omitted here and left
    untouched in any identity-candidate file. The finished graph is stored as
    run ``github-multi-ego`` and reused instead of calling the API again.
    """
    requested = _normalize(handles)
    if not requested:
        raise ValueError("handles must include a public GitHub login")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    config = _request_config(requested, max_pages)
    cached = _cached_union(config)
    if cached is not None:
        return cached
    from osi.store import create_run, save_graph

    combined = nx.Graph()
    skipped: list[str] = []
    for handle in requested:
        piece = _cached_piece(handle, max_pages)
        if piece is None:
            piece = _fetch_one(handle, max_pages)
            _store_piece(handle, piece)
        if piece.graph.get("skipped"):
            skipped.append(str(piece.graph["skipped"]))
        _merge(combined, piece)
    # Shared neighbors are the edges a single star does not have. Save only the finished union.
    combined.graph["handles"] = config["handles"]
    combined.graph["max_pages"] = max_pages
    save_graph(MULTI_EGO_RUN, MULTI_EGO_LAYER, combined)
    note = "skipped zero-follower logins: " + ", ".join(skipped) if skipped else ""
    create_run("github", config, notes=note, run_id=MULTI_EGO_RUN)
    return combined


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
