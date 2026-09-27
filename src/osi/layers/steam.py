"""Steam friend layer from SteamGPT.

SteamGPT is the anonymous service behind the MCP server at
https://steamgpt.net/mcp. These calls use its JSON routes, which are the same
data those tools return. No Steam Web API key is read or sent.
"""

from __future__ import annotations

import sys
import time
from typing import Any
from urllib.parse import quote

import networkx as nx

from osi.limits import STEAM_MAX_FRIENDS, get_json

# One request per second. SteamGPT's own fair-use cap is 120 per minute.
_PAUSE_SECONDS = 1.0
_BASE = "https://steamgpt.net"
_last_request = 0.0


def _throttled(url: str) -> Any:
    """GET one SteamGPT URL, waiting so requests stay a second apart."""
    global _last_request
    if _last_request:
        delay = _PAUSE_SECONDS - (time.monotonic() - _last_request)
        if delay > 0:
            time.sleep(delay)
    payload = get_json(url)
    _last_request = time.monotonic()
    return payload


def _read(url: str) -> dict | None:
    """Return a success payload. A private profile or a failed call becomes None."""
    try:
        payload = _throttled(url)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"note: steam request failed: {error}", file=sys.stderr)
        return None
    if not isinstance(payload, dict) or payload.get("result") == "error":
        # SteamGPT reports a private or unknown player as an error result, not a guess.
        message = payload.get("message") if isinstance(payload, dict) else "bad payload"
        print(f"note: steam request failed: {message}", file=sys.stderr)
        return None
    data = payload.get("data")
    return data if isinstance(data, dict) else None


def resolve_vanity(vanity: str, api_key: str | None = None) -> str | None:
    """Convert a custom Steam URL name to a 64-bit SteamID.

    Uses SteamGPT's identity route, the same lookup as ISteamUser/ResolveVanityURL.
    A value that is already 17 digits is returned as-is. api_key is ignored:
    SteamGPT does not accept a Valve key, and this function never sends one.
    """
    del api_key
    text = str(vanity).strip()
    if not text:
        return None
    # A steamid64 does not need a vanity lookup. Anything else can be a custom URL.
    if text.isdigit() and len(text) >= 17:
        return text
    data = _read(f"{_BASE}/identity/{quote(text)}.json")
    if not data or not data.get("resolved"):
        print(f"note: steam vanity {text} did not resolve", file=sys.stderr)
        return None
    steam_id = str(data.get("steamid64") or "")
    return steam_id or None


def fetch_steam_friends(steam_id: str, api_key: str | None = None) -> list[str]:
    """Public friend SteamIDs for one player.

    SteamGPT's friend route is the public friend-graph snapshot, the stand-in
    for ISteamUser/GetFriendList. A private profile or a failed call returns [].
    api_key is ignored and never sent.
    """
    del api_key
    text = str(steam_id).strip()
    if not text:
        return []
    data = _read(f"{_BASE}/friends/{quote(text)}.json?detail=short&limit={STEAM_MAX_FRIENDS}")
    if not data:
        print(f"note: no public friends for {text}; the profile may be private", file=sys.stderr)
        return []
    friends = data.get("friends") or []
    if not isinstance(friends, list):
        print(f"note: no public friends for {text}; the profile may be private", file=sys.stderr)
        return []
    # detail=short is a list of steamid64 strings. A richer row still carries steamid64.
    found: list[str] = []
    for friend in friends[:STEAM_MAX_FRIENDS]:
        if isinstance(friend, str) and friend:
            found.append(friend)
        elif isinstance(friend, dict) and friend.get("steamid64"):
            found.append(str(friend["steamid64"]))
    if not found:
        print(f"note: no public friends for {text}; the profile may be private", file=sys.stderr)
    return found


def fetch_steam_games(steam_id: str, api_key: str | None = None) -> list[dict[str, Any]]:
    """Owned games as {appid, name, playtime_forever}.

    SteamGPT does not publish GetOwnedGames. A private library and this missing
    route both return []. api_key is ignored and never sent.
    """
    del api_key
    print(
        f"note: no public game library for {steam_id}; owned games left empty",
        file=sys.stderr,
    )
    return []


def build_steam_layer(identity_map: dict, api_key: str | None = None) -> nx.Graph:
    """One undirected friend graph for every person who has a steam handle.

    The center node is ``person_id|steam``. Each friend is a steamid64, with a
    mutual edge of weight 1. Two people who share a friend meet at that node.
    Owned games are not added: SteamGPT has no co-ownership list to build from.
    """
    graph = nx.Graph()
    seen: set[str] = set()
    for person, accounts in identity_map.items():
        if not isinstance(accounts, dict):
            continue
        handle = accounts.get("steam")
        if not handle:
            continue
        handle = str(handle).strip()
        if not handle or handle in seen:
            continue
        seen.add(handle)
        steam_id = resolve_vanity(handle, api_key)
        if not steam_id:
            print(f"note: steam handle {handle} for {person} skipped", file=sys.stderr)
            continue
        friends = fetch_steam_friends(steam_id, api_key)
        # The call keeps the spec's game step. The list is empty, so no game nodes are added.
        fetch_steam_games(steam_id, api_key)
        center = f"{person}|steam"
        graph.add_node(
            center,
            platform="steam",
            label=handle,
            steam_id=steam_id,
            public=True,
            center=True,
        )
        for friend_id in friends:
            if friend_id == steam_id:
                continue
            if friend_id not in graph:
                graph.add_node(
                    friend_id,
                    platform="steam",
                    label=friend_id,
                    steam_id=friend_id,
                    public=True,
                    center=False,
                )
            graph.add_edge(center, friend_id, relation="friend", direction="mutual", weight=1.0, layer="friend")
    return graph
