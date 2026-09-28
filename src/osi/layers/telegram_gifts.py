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

# One Telegram API call per 2.4 seconds. A cache hit does not count.
_PAUSE_SECONDS = 2.4
_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _ROOT / "data" / "telegram_cache"
_CACHE_PATH = _CACHE_DIR / "users.json"
_LEGACY_USER_CACHE = _ROOT / "data" / "telegram_username_cache.json"
_cache: dict | None = None
_last_resolution = 0.0

# Seeds that Telegram accepted on the previous run, plus the three new names.
# tassagency, rt_com, nasa, and vk never resolved, so they are not seeds.
CRAWL_SEEDS = (
    "durov",
    "telegram",
    "mygefs",
    "aaaappp21",
    "Pantelidass",
    "thegreyestofalltime",
    "meduzalive",
    "bbcrussian",
    "currenttime",
    "rian_ru",
    "wikipedia",
    "github",
    "discord",
    "spotify",
    "steam",
    "yandex",
    "snowden",
    "lexfridman",
    "vitalikbuterin",
)
# Five waves: the seeds, then four expansions. Depth 5 is recorded but not fetched.
MAX_ITERATIONS = 5
MAX_ACCOUNTS = 500


def _username_cache() -> dict:
    """Load the username cache once per process."""
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.is_file():
            _cache.update(json.loads(_CACHE_PATH.read_text(encoding="utf-8")))
        # The first crawl stored ids beside data/, not inside telegram_cache/.
        if _CACHE_PATH == _CACHE_DIR / "users.json" and _LEGACY_USER_CACHE.is_file():
            legacy = json.loads(_LEGACY_USER_CACHE.read_text(encoding="utf-8"))
            for key, username in legacy.items():
                _cache.setdefault(str(key), username)
    return _cache


def _write_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _CACHE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(_CACHE_PATH)


async def _pace_resolution() -> None:
    """Wait so the next API call starts at least 2.4 seconds after the previous one."""
    if _last_resolution:
        remaining = _PAUSE_SECONDS - (time.monotonic() - _last_resolution)
        if remaining > 0:
            await asyncio.sleep(remaining)


def _mark_api_call() -> None:
    global _last_resolution
    _last_resolution = time.monotonic()


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
    # The clock starts when the lookup finishes, so the next call cannot start early.
    _mark_api_call()
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


def _dedup_key(account: str) -> str:
    # Usernames are case-insensitive. Numeric accounts stay distinct.
    if account.startswith("id:"):
        return account
    return account.casefold()


def _account_slug(account: str) -> str:
    if account.startswith("id:"):
        return "id_" + account.split(":", 1)[1]
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in account).casefold()


def _gift_path(account: str) -> Path:
    return _CACHE_PATH.parent / "gifts" / f"{_account_slug(account)}.json"


def _state_path() -> Path:
    return _CACHE_PATH.parent / "state.json"


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _gift_payload(gift: dict) -> dict:
    date = gift.get("date")
    # datetime is not JSON. Store the ISO string and keep the other fields.
    if hasattr(date, "isoformat"):
        date = date.isoformat()
    return {
        "sender_id": gift.get("sender_id"),
        "gift_title": gift.get("gift_title"),
        "gift_value": gift.get("gift_value"),
        "date": date,
    }


def _peer_for(account: str):
    # id:123 is a numeric peer. Anything else is a public username.
    if account.startswith("id:"):
        return int(account.split(":", 1)[1])
    return account


def _node_label(account: str) -> str:
    if account.startswith("id:"):
        return account.split(":", 1)[1]
    return account


def _permanent_error(message: str) -> bool:
    # These replies will not succeed on a retry. Connection drops will.
    markers = (
        "No user has",
        "Nobody is using",
        "unacceptable",
        "Cannot find any entity",
    )
    return any(marker in message for marker in markers)


def _load_crawl_state(seeds: tuple[str, ...] | list[str], max_accounts: int, max_iterations: int) -> dict:
    state = _read_json(_state_path()) or {"queue": [], "seen": {}, "done": [], "retries": {}}
    state.setdefault("queue", [])
    state.setdefault("seen", {})
    state.setdefault("done", [])
    state.setdefault("retries", {})
    for seed in seeds:
        key = _dedup_key(seed)
        # A later run can add a seed without throwing away accounts already fetched.
        if key in state["seen"]:
            continue
        state["seen"][key] = 0
        state["queue"].append({"account": seed, "depth": 0})
    state["max_accounts"] = max_accounts
    state["max_iterations"] = max_iterations
    return state


def _save_crawl_state(state: dict) -> None:
    _write_json(_state_path(), state)


async def _gifts_for_account(client, account: str) -> tuple[list[dict], bool]:
    """Return cached gifts, or fetch and store them.

    The bool is True when this call performed an API request.
    """
    path = _gift_path(account)
    cached = _read_json(path)
    if cached is not None:
        return list(cached.get("gifts") or []), False
    await _pace_resolution()
    try:
        gifts = [_gift_payload(gift) for gift in await fetch_user_gifts(client, _peer_for(account))]
    except Exception as error:
        _mark_api_call()
        message = f"{type(error).__name__}: {error}"
        if _permanent_error(message):
            _write_json(path, {"account": account, "ok": False, "error": message, "gifts": []})
        raise
    _mark_api_call()
    _write_json(path, {"account": account, "ok": True, "error": None, "gifts": gifts})
    return gifts, True


async def crawl_gift_network(
    client,
    seeds: tuple[str, ...] | list[str] | None = None,
    *,
    max_accounts: int = MAX_ACCOUNTS,
    max_iterations: int = MAX_ITERATIONS,
) -> nx.DiGraph:
    """Expand from seeds by fetching the public gifts of each discovered sender.

    Depth 0 is the seed wave. An account at depth 5 is not fetched.
    At most `max_accounts` profiles are fetched. Restarting reads data/telegram_cache/.
    """
    selected = list(seeds if seeds is not None else CRAWL_SEEDS)
    state = _load_crawl_state(selected, max_accounts, max_iterations)
    done = set(state["done"])
    while state["queue"] and len(done) < max_accounts:
        item = state["queue"].pop(0)
        account = str(item["account"])
        depth = int(item["depth"])
        key = _dedup_key(account)
        if key in done or depth >= max_iterations:
            continue
        try:
            gifts, requested = await _gifts_for_account(client, account)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            state["retries"][key] = int(state["retries"].get(key, 0)) + 1
            # A flaky connection is retried once. A missing username is final.
            if _permanent_error(message) or state["retries"][key] >= 2:
                done.add(key)
                state["done"] = list(done)
                print(f"warning: gifts for {account} not loaded: {message}", file=sys.stderr)
            else:
                state["queue"].append(item)
            _save_crawl_state(state)
            continue
        done.add(key)
        state["done"] = list(done)
        origin = "api" if requested else "cache"
        print(
            f"crawl depth={depth} account={account} gifts={len(gifts)} "
            f"via={origin} fetched={len(done)}/{max_accounts} queue={len(state['queue'])}",
            file=sys.stderr,
        )
        child_depth = depth + 1
        for gift in gifts:
            sender_id = gift.get("sender_id")
            if sender_id is None:
                continue
            username = await resolve_sender(client, sender_id)
            child = username or f"id:{int(sender_id)}"
            child_key = _dedup_key(child)
            if child_key in state["seen"] or child_depth >= max_iterations:
                continue
            # The cap counts fetched profiles plus the ones already waiting.
            if len(done) + len(state["queue"]) >= max_accounts:
                break
            state["seen"][child_key] = child_depth
            state["queue"].append({"account": child, "depth": child_depth})
        _save_crawl_state(state)
    graph = _graph_from_gift_cache()
    _write_json(_CACHE_PATH.parent / "graph.json", nx.node_link_data(graph, edges="links"))
    return graph


def _graph_from_gift_cache() -> nx.DiGraph:
    """Build the directed graph from every cached gift file."""
    graph = nx.DiGraph()
    graph.graph["directed"] = True
    graph.graph["layer"] = "telegram_gifts"
    graph.graph["edge_direction"] = "sender -> recipient"
    gift_dir = _CACHE_PATH.parent / "gifts"
    if not gift_dir.is_dir():
        return graph
    usernames = _username_cache()
    for path in sorted(gift_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("ok"):
            continue
        account = str(record.get("account") or path.stem)
        gifts = list(record.get("gifts") or [])
        target = _telegram_node(_node_label(account))
        for gift in gifts:
            sender_id = gift.get("sender_id")
            if sender_id is None:
                continue
            username = usernames.get(str(int(sender_id)))
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


def run_crawler() -> nx.DiGraph:
    """Crawl public gift profiles and save the directed graph."""
    client = _client()

    async def run() -> nx.DiGraph:
        await client.connect()
        if not await client.is_user_authorized():
            raise SystemExit("Telegram session is not logged in. Run python3 test_telegram.py first.")
        graph = await crawl_gift_network(client)
        await client.disconnect()
        return graph

    graph = asyncio.run(run())
    print(f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    seed_nodes = {_telegram_node(seed) for seed in CRAWL_SEEDS}
    shared: dict[str, set[str]] = {}
    for sender, recipient in graph.edges:
        if recipient not in seed_nodes:
            continue
        shared.setdefault(sender, set()).add(recipient)
    cross = sum(1 for recipients in shared.values() if len(recipients) > 1)
    print(f"senders who gifted more than one seed: {cross}")
    print("Top senders:")
    for node in sorted(graph.nodes, key=lambda name: graph.out_degree(name, weight="weight"), reverse=True)[:10]:
        print(f"  {node}: sent {graph.out_degree(node, weight='weight')}")
    from osi.store import create_run, save_graph

    # The saved run stays directed. Analysis loads this id instead of crawling again.
    create_run(
        "telegram_gifts",
        {
            "source": "telegram_gifts",
            "layer": "telegram_gifts",
            "directed": True,
            "edge_direction": "sender -> recipient",
            "max_accounts": MAX_ACCOUNTS,
            "max_iterations": MAX_ITERATIONS,
        },
        "expanded gift crawl",
        run_id="telegram-gifts-crawl-v1",
    )
    save_graph("telegram-gifts-crawl-v1", "telegram_gifts", graph)
    return graph


if __name__ == "__main__":
    run_crawler()
