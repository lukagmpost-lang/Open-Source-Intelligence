"""Telegram star-gift layer.

Fetches gifts pinned on a profile and turns them into a directed
sender -> recipient graph. Hidden senders stay anonymous.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import networkx as nx
from telethon import TelegramClient
from telethon.tl.functions.payments import GetSavedStarGiftsRequest
from telethon.utils import get_peer_id

# Public accounts. The first three are the required seeds.
# The other seventeen are public newsrooms, companies, and public figures.
SEED_USERNAMES = (
    "durov",
    "telegram",
    "mygefs",
    "meduzalive",
    "bbcrussian",
    "currenttime",
    "rian_ru",
    "tassagency",
    "rt_com",
    "nasa",
    "wikipedia",
    "github",
    "discord",
    "spotify",
    "steam",
    "vk",
    "yandex",
    "snowden",
    "lexfridman",
    "vitalikbuterin",
)

# One entity lookup per two seconds. A cache hit does not count.
_PAUSE_SECONDS = 2.0
_CACHE_PATH = Path(__file__).resolve().parents[3] / "data" / "telegram_username_cache.json"
_cache: dict | None = None
_last_resolution = 0.0


def _username_cache() -> dict:
    """Load the username cache once per process."""
    global _cache
    if _cache is None:
        if _CACHE_PATH.is_file():
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _cache = {}
    return _cache


def _write_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(_CACHE_PATH)


async def _pace_resolution() -> None:
    """Wait until two seconds have passed since the previous entity lookup."""
    global _last_resolution
    if _last_resolution:
        remaining = _PAUSE_SECONDS - (time.monotonic() - _last_resolution)
        if remaining > 0:
            await asyncio.sleep(remaining)


def _telegram_node(label) -> str:
    return f"telegram:{label}"


async def resolve_sender(client, sender_id):
    """Resolve a numeric Telegram user ID to a username.

    Returns username string or None if the user has no public username.
    """
    key = str(int(sender_id))
    cache = _username_cache()
    if key in cache:
        # None is stored too, so a hidden or nameless account is not requested again.
        return cache[key]
    await _pace_resolution()
    try:
        entity = await client.get_entity(int(sender_id))
    except Exception:
        username = None
    else:
        username = getattr(entity, "username", None) or None
    global _last_resolution
    # The clock starts when the lookup finishes, so the next one cannot start early.
    _last_resolution = time.monotonic()
    cache[key] = username
    _write_cache(cache)
    return username


async def fetch_user_gifts(client, username, *, exclude_unsaved=True):
    """Fetch gifts received by a user that are displayed on their profile.

    peer=username fetches gifts received by that user, not the logged-in account.
    exclude_unsaved=True returns only gifts pinned to the profile.
    from_id on each gift is the sender. None means anonymous.
    """
    # Telethon 1.45.0 requires offset; "" is the first page.
    # The client resolves a username through get_input_entity before sending.
    result = await client(GetSavedStarGiftsRequest(
        peer=username,
        offset="",
        limit=100,
        exclude_unsaved=exclude_unsaved,
    ))
    gifts = []
    for gift in result.gifts:
        from_id = getattr(gift, "from_id", None)
        # from_id is a Peer. Hidden senders omit it, so sender_id stays None.
        sender_id = None if from_id is None else get_peer_id(from_id)
        gift_obj = getattr(gift, "gift", None)
        gifts.append({
            "sender_id": sender_id,
            "gift_title": getattr(gift_obj, "title", None) if gift_obj is not None else None,
            "gift_value": getattr(gift, "convert_stars", None),
            "date": getattr(gift, "date", None),
        })
    return gifts


async def build_gift_graph(client, usernames):
    """Fetch gifts for each username and build a directed graph.

    Edges: sender -> recipient, weight = number of gifts.
    A public username is the node id. A numeric id is the fallback.
    """
    received: dict[str, list[dict]] = {}
    sender_ids: set[int] = set()
    for user in usernames:
        try:
            gifts = await fetch_user_gifts(client, user)
        except Exception as error:
            print(f"warning: gifts for {user} not loaded: {error}", file=sys.stderr)
            continue
        print(f"gifts {user}: {len(gifts)}", file=sys.stderr)
        received[user] = gifts
        for gift in gifts:
            if gift["sender_id"] is not None:
                sender_ids.add(int(gift["sender_id"]))

    resolved: dict[int, str | None] = {}
    ordered_ids = sorted(sender_ids)
    print(f"resolving {len(ordered_ids)} senders", file=sys.stderr)
    for index, sender_id in enumerate(ordered_ids, start=1):
        resolved[sender_id] = await resolve_sender(client, sender_id)
        # Cache hits return immediately, so this marks real progress through the paced lookups.
        if index == 1 or index % 25 == 0 or index == len(ordered_ids):
            print(f"resolved {index}/{len(ordered_ids)}", file=sys.stderr)

    graph = nx.DiGraph()
    # Stored with the layer so later runs know the edges point from sender to recipient.
    graph.graph["directed"] = True
    graph.graph["layer"] = "telegram_gifts"
    graph.graph["edge_direction"] = "sender -> recipient"
    for user, gifts in received.items():
        target = _telegram_node(user)
        for gift in gifts:
            sender_id = gift["sender_id"]
            if sender_id is None:
                continue
            username = resolved.get(int(sender_id))
            # Prefer the public username. Keep the numeric id when Telegram has none.
            source = _telegram_node(username or sender_id)
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)
        graph.add_node(target, gifts_received=len(gifts))
    return graph


def _load_api_credentials() -> tuple[int, str]:
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # A value already exported in the shell wins.
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    api_id = os.environ.get("TELEGRAM_API_ID", "")
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    if not api_id.isdigit() or not api_hash:
        raise SystemExit("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set")
    return int(api_id), api_hash


def _client() -> TelegramClient:
    api_id, api_hash = _load_api_credentials()
    session_path = Path.home() / ".osi" / "telegram.session"
    client = TelegramClient(str(session_path), api_id, api_hash, timeout=20, connection_retries=2)
    original_set_dc = client.session.set_dc

    def set_dc(dc_id, server_address, port):
        # Port 443 on these data centers is answered with HTTP. 5222 speaks MTProto.
        return original_set_dc(dc_id, server_address, 5222)

    client.session.set_dc = set_dc
    if client.session.server_address and client.session.port != 5222:
        original_set_dc(client.session.dc_id, client.session.server_address, 5222)
    return client


def build_telegram_gifts_layer(usernames: tuple[str, ...] | None = None) -> nx.DiGraph:
    """Build the directed gift graph from the saved Telegram session."""
    seeds = usernames if usernames is not None else SEED_USERNAMES
    client = _client()

    async def run() -> nx.DiGraph:
        await client.connect()
        if not await client.is_user_authorized():
            raise SystemExit("Telegram session is not logged in. Run python3 test_telegram.py first.")
        graph = await build_gift_graph(client, seeds)
        await client.disconnect()
        return graph

    return asyncio.run(run())
