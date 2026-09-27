import osi.layers.steam as steam


def _install(monkeypatch):
    steam._last_request = 0.0
    calls = []
    slept = []

    def fake_get(url, headers=None):
        calls.append(url)
        if url.endswith("/identity/gaben.json"):
            return {"result": "success", "data": {"steamid64": "76561197968052866", "resolved": True}}
        if url.endswith("/identity/robinwalker.json"):
            return {"result": "success", "data": {"steamid64": "76561197960435530", "resolved": True}}
        if url.endswith("/identity/missing.json"):
            return {"result": "error", "message": "player_not_found"}
        if "/friends/76561197968052866.json" in url:
            return {"result": "success", "data": {"friends": ["76561197960435530", "76561198000000000"]}}
        if "/friends/76561197960435530.json" in url:
            return {"result": "success", "data": {"friends": ["76561198000000000"]}}
        if "/friends/private.json" in url:
            raise RuntimeError("upstream HTTP 401 for friends")
        raise AssertionError(url)

    monkeypatch.setattr(steam, "get_json", fake_get)
    monkeypatch.setattr(steam.time, "sleep", lambda seconds: slept.append(seconds))
    return calls, slept


def test_shared_friend_links_two_profiles_and_a_key_is_never_sent(monkeypatch):
    calls, slept = _install(monkeypatch)
    graph = steam.build_steam_layer(
        {
            "gabe_newell": {"steam": "gaben"},
            "robin_walker": {"steam": "robinwalker"},
            "nobody": {"steam": "missing"},
            "linus_torvalds": {"github": "torvalds"},
        },
        api_key="do-not-send",
    )
    assert "gabe_newell|steam" in graph
    assert graph.nodes["gabe_newell|steam"]["steam_id"] == "76561197968052866"
    # The shared friend is one node with an edge to each center.
    assert graph.degree("76561198000000000") == 2
    assert graph.edges["gabe_newell|steam", "76561198000000000"]["direction"] == "mutual"
    assert "nobody|steam" not in graph
    assert all("do-not-send" not in url and "key=" not in url for url in calls)
    assert slept
    assert all(seconds <= 1.0 for seconds in slept)


def test_private_friends_and_missing_games_return_empty(monkeypatch, capsys):
    _install(monkeypatch)
    assert steam.fetch_steam_friends("private") == []
    assert steam.fetch_steam_games("76561197968052866", api_key="secret") == []
    error = capsys.readouterr().err
    assert "private" in error
    assert "owned games left empty" in error
    assert "secret" not in error


def test_a_steamid64_skips_the_vanity_lookup(monkeypatch):
    calls, _slept = _install(monkeypatch)
    assert steam.resolve_vanity("76561197968052866") == "76561197968052866"
    assert calls == []
