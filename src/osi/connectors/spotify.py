"""Spotify playlists the caller is allowed to read with their own token."""

from __future__ import annotations

import os

from osi.graph import Graph
from osi.limits import SPOTIFY_MAX_PLAYLISTS, SPOTIFY_MAX_TRACKS, get_json


def _headers() -> dict[str, str]:
    token = os.environ.get("SPOTIFY_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SPOTIFY_TOKEN is required; this reads only the token owner's playlists")
    return {"Authorization": f"Bearer {token}"}


def fetch_spotify() -> Graph:
    headers = _headers()
    me = get_json("https://api.spotify.com/v1/me", headers)
    user_id = me.get("id")
    if not user_id:
        raise RuntimeError("Spotify token did not resolve to an account")
    graph = Graph()
    root = graph.add_node("spotify", user_id, me.get("display_name") or user_id)
    playlists = get_json(
        f"https://api.spotify.com/v1/me/playlists?limit={SPOTIFY_MAX_PLAYLISTS}",
        headers,
    )
    for playlist in (playlists.get("items") or [])[:SPOTIFY_MAX_PLAYLISTS]:
        if not playlist or not playlist.get("public"):
            continue
        playlist_id = playlist.get("id")
        if not playlist_id:
            continue
        graph.add_node("spotify", f"playlist/{playlist_id}", playlist.get("name") or playlist_id, {"kind": "playlist"})
        graph.add_edge(root.id, f"spotify:playlist/{playlist_id}", "playlist", 1.0)
        tracks = get_json(
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit={SPOTIFY_MAX_TRACKS}",
            headers,
        )
        for item in (tracks.get("items") or [])[:SPOTIFY_MAX_TRACKS]:
            track = (item or {}).get("track") or {}
            track_id = track.get("id")
            if not track_id:
                continue
            graph.add_node("spotify", f"track/{track_id}", track.get("name") or track_id, {"kind": "track"})
            graph.add_edge(f"spotify:playlist/{playlist_id}", f"spotify:track/{track_id}", "playlist", 1.0)
    return graph
