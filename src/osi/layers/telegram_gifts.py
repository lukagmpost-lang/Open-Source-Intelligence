"""Telegram star-gift layer.

Fetches gifts saved on a profile and turns them into a directed
sender -> recipient graph. Hidden senders stay anonymous.
"""

from telethon.tl.functions.payments import GetSavedStarGiftsRequest
from telethon.utils import get_peer_id


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
    for g in result.gifts:
        from_id = getattr(g, "from_id", None)
        # from_id is a Peer. Hidden senders omit it, so sender_id stays None.
        sender_id = None if from_id is None else get_peer_id(from_id)
        gift_obj = getattr(g, "gift", None)
        gifts.append({
            "sender_id": sender_id,
            "gift_title": getattr(gift_obj, "title", None) if gift_obj is not None else None,
            "gift_value": getattr(g, "convert_stars", None),
            "date": getattr(g, "date", None),
        })
    return gifts


async def build_gift_graph(client, usernames):
    """Fetch gifts for each username and build a directed graph.

    Edges: sender -> recipient, weight = number of gifts.
    """
    import networkx as nx

    graph = nx.DiGraph()
    for user in usernames:
        gifts = await fetch_user_gifts(client, user)
        for gift in gifts:
            if gift["sender_id"] is None:
                continue
            source = f"telegram:{gift['sender_id']}"
            target = f"telegram:{user}"
            # add_edge overwrites weight, so repeat gifts from one sender must accumulate.
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
            else:
                graph.add_edge(source, target, weight=1)
        graph.add_node(f"telegram:{user}", gifts_received=len(gifts))
    return graph
