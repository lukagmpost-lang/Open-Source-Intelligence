"""Steam Web API for profiles the player has set to public."""

from __future__ import annotations

import os

from osi.graph import Graph
from osi.limits import STEAM_MAX_FRIENDS, STEAM_MAX_GAMES, get_json


def fetch_steam(steam_id: str) -> Graph:
    steam_id = steam_id.strip()
    key = os.environ.get("STEAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("STEAM_API_KEY is required")
    if not steam_id.isdigit():
        raise ValueError("steam_id must be a public 64-bit SteamID")
    summary = get_json(
        "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/"
        f"?key={key}&steamids={steam_id}"
    )
    players = ((summary.get("response") or {}).get("players")) or []
    if not players or players[0].get("communityvisibilitystate") != 3:
        raise RuntimeError("Steam profile is not public")
    player = players[0]
    graph = Graph()
    root = graph.add_node("steam", steam_id, player.get("personaname") or steam_id)
    friends = get_json(
        "https://api.steampowered.com/ISteamUser/GetFriendList/v1/"
        f"?key={key}&steamid={steam_id}&relationship=friend"
    )
    friend_list = ((friends.get("friendslist") or {}).get("friends")) or []
    for friend in friend_list[:STEAM_MAX_FRIENDS]:
        friend_id = str(friend.get("steamid") or "")
        if not friend_id:
            continue
        graph.add_node("steam", friend_id, friend_id)
        graph.add_edge(root.id, f"steam:{friend_id}", "friend", 1.0)
    games = get_json(
        "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
        f"?key={key}&steamid={steam_id}&include_appinfo=1&include_played_free_games=0"
    )
    for game in (((games.get("response") or {}).get("games")) or [])[:STEAM_MAX_GAMES]:
        appid = game.get("appid")
        if appid is None:
            continue
        name = game.get("name") or str(appid)
        graph.add_node("steam", f"app/{appid}", name, {"kind": "game"})
        graph.add_edge(root.id, f"steam:app/{appid}", "library", 1.0)
    return graph
