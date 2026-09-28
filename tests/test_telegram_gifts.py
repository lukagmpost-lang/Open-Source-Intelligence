"""Gift-layer tests that never open a Telegram connection."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from telethon.tl.functions.payments import GetSavedStarGiftsRequest
from telethon.tl.types import PeerUser

from osi.layers import telegram_gifts
from osi.layers.telegram_gifts import build_gift_graph, fetch_user_gifts, resolve_sender


@pytest.fixture(autouse=True)
def isolated_username_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(telegram_gifts, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(telegram_gifts, "_cache", None)
    monkeypatch.setattr(telegram_gifts, "_last_resolution", 0.0)
    # The live pace is 2.4 seconds. Tests should not wait.
    monkeypatch.setattr(telegram_gifts, "_PAUSE_SECONDS", 0.0)


class _Client:
    def __init__(self, gifts):
        self.gifts = gifts
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return SimpleNamespace(gifts=self.gifts)


def test_fetch_user_gifts_uses_saved_star_gifts_request():
    when = datetime(2026, 1, 2, tzinfo=timezone.utc)
    client = _Client([
        SimpleNamespace(
            from_id=PeerUser(user_id=42),
            gift=SimpleNamespace(title="Plush Pepe"),
            convert_stars=50,
            date=when,
        ),
        SimpleNamespace(
            from_id=None,
            gift=SimpleNamespace(title="Anonymous Star"),
            convert_stars=10,
            date=None,
        ),
    ])

    gifts = asyncio.run(fetch_user_gifts(client, "alice"))

    request = client.requests[0]
    assert isinstance(request, GetSavedStarGiftsRequest)
    # peer is the target profile, so the gifts are ones that user received.
    assert request.peer == "alice"
    assert request.offset == ""
    assert request.limit == 100
    assert request.exclude_unsaved is True
    assert gifts == [
        {
            "sender_id": 42,
            "gift_title": "Plush Pepe",
            "gift_value": 50,
            "date": when,
        },
        {
            "sender_id": None,
            "gift_title": "Anonymous Star",
            "gift_value": 10,
            "date": None,
        },
    ]


def test_fetch_user_gifts_can_include_gifts_not_pinned_to_profile():
    client = _Client([])

    asyncio.run(fetch_user_gifts(client, "durov", exclude_unsaved=False))

    # False leaves the flag unset, so unsaved gifts are included too.
    assert client.requests[0].exclude_unsaved is False
    assert client.requests[0].peer == "durov"


def test_build_gift_graph_counts_gifts_and_skips_anonymous_senders():
    client = _Client([
        SimpleNamespace(from_id=PeerUser(user_id=7), gift=SimpleNamespace(title="A"), convert_stars=1, date=None),
        SimpleNamespace(from_id=PeerUser(user_id=7), gift=SimpleNamespace(title="B"), convert_stars=1, date=None),
        SimpleNamespace(from_id=None, gift=SimpleNamespace(title="C"), convert_stars=1, date=None),
    ])

    graph = asyncio.run(build_gift_graph(client, ["bob"]))

    assert graph.nodes["telegram:bob"]["gifts_received"] == 3
    assert graph["telegram:7"]["telegram:bob"]["weight"] == 2
    assert "telegram:None" not in graph
    assert graph.graph["directed"] is True
    assert graph.graph["edge_direction"] == "sender -> recipient"


class _ResolvingClient(_Client):
    def __init__(self, gifts, names):
        super().__init__(gifts)
        self.names = names
        self.lookups = []

    async def get_entity(self, sender_id):
        self.lookups.append(int(sender_id))
        if sender_id not in self.names:
            raise ValueError("unknown")
        return SimpleNamespace(username=self.names[sender_id])


def test_build_gift_graph_uses_username_and_caches_the_lookup():
    client = _ResolvingClient(
        [
            SimpleNamespace(from_id=PeerUser(user_id=7), gift=SimpleNamespace(title="A"), convert_stars=1, date=None),
            SimpleNamespace(from_id=PeerUser(user_id=7), gift=SimpleNamespace(title="B"), convert_stars=1, date=None),
            SimpleNamespace(from_id=PeerUser(user_id=8), gift=SimpleNamespace(title="C"), convert_stars=1, date=None),
        ],
        {7: "alice", 8: None},
    )

    graph = asyncio.run(build_gift_graph(client, ["bob"]))

    assert graph["telegram:alice"]["telegram:bob"]["weight"] == 2
    assert graph["telegram:8"]["telegram:bob"]["weight"] == 1
    first_lookups = list(client.lookups)
    asyncio.run(build_gift_graph(client, ["bob"]))
    # The second build reads the cache and does not call Telegram again.
    assert client.lookups == first_lookups


def test_resolve_sender_returns_none_without_a_public_username():
    class _ClientWithoutName:
        def __init__(self):
            self.calls = 0

        async def get_entity(self, sender_id):
            self.calls += 1
            return SimpleNamespace(username=None)

    client = _ClientWithoutName()
    assert asyncio.run(resolve_sender(client, 9)) is None
    assert asyncio.run(resolve_sender(client, 9)) is None
    assert client.calls == 1


class _CrawlClient:
    def __init__(self, profiles, names):
        self.profiles = profiles
        self.names = names
        self.fetches = []

    async def __call__(self, request):
        self.fetches.append(request.peer)
        sender_ids = self.profiles.get(request.peer, [])
        gifts = [
            SimpleNamespace(
                from_id=PeerUser(user_id=sender_id),
                gift=SimpleNamespace(title=None),
                convert_stars=None,
                date=None,
            )
            for sender_id in sender_ids
        ]
        return SimpleNamespace(gifts=gifts)

    async def get_entity(self, sender_id):
        if sender_id not in self.names:
            raise ValueError("missing")
        return SimpleNamespace(username=self.names[sender_id])


def test_crawl_expands_senders_and_resumes_from_cache():
    client = _CrawlClient(
        {"durov": [1], "alice": [2], "bob": []},
        {1: "alice", 2: "bob"},
    )

    graph = asyncio.run(telegram_gifts.crawl_gift_network(
        client,
        ["durov"],
        max_accounts=10,
        max_iterations=5,
    ))

    assert client.fetches == ["durov", "alice", "bob"]
    assert graph["telegram:alice"]["telegram:durov"]["weight"] == 1
    assert graph["telegram:bob"]["telegram:alice"]["weight"] == 1
    assert graph.graph["directed"] is True
    client.fetches.clear()
    again = asyncio.run(telegram_gifts.crawl_gift_network(
        client,
        ["durov"],
        max_accounts=10,
        max_iterations=5,
    ))
    # The second pass reads data/telegram_cache and does not call Telegram.
    assert client.fetches == []
    assert again.number_of_edges() == graph.number_of_edges()


def test_crawl_stops_after_the_account_cap():
    client = _CrawlClient({"durov": [1], "alice": [2]}, {1: "alice", 2: "bob"})

    asyncio.run(telegram_gifts.crawl_gift_network(
        client,
        ["durov"],
        max_accounts=1,
        max_iterations=5,
    ))

    assert client.fetches == ["durov"]


def test_crawl_stops_at_iteration_depth():
    client = _CrawlClient(
        {"durov": [1], "alice": [2], "bob": [3]},
        {1: "alice", 2: "bob", 3: "cara"},
    )

    asyncio.run(telegram_gifts.crawl_gift_network(
        client,
        ["durov"],
        max_accounts=10,
        max_iterations=2,
    ))

    # Depth 0 and 1 are fetched. bob is depth 2, which is past the second wave.
    assert client.fetches == ["durov", "alice"]
