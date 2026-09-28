"""Gift-layer tests that never open a Telegram connection."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from telethon.tl.functions.payments import GetSavedStarGiftsRequest
from telethon.tl.types import PeerUser

from osi.layers.telegram_gifts import build_gift_graph, fetch_user_gifts


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
