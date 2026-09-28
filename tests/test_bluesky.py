import osi.layers.bluesky as bluesky


def _install(monkeypatch):
    bluesky._last_request = 0.0
    bluesky._host_index = 0
    calls = []
    slept = []

    def fake_get(url, headers=None):
        calls.append(url)
        if "public.api.bsky.app" in url and "actor=switch.bsky.social" in url:
            raise RuntimeError("upstream HTTP 403 for https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile")
        if "actor=missing.bsky.social" in url:
            raise RuntimeError("upstream HTTP 400 for getProfile")
        if "getProfile" in url and "actor=jay.bsky.team" in url:
            return {
                "handle": "jay.bsky.team",
                "did": "did:jay",
                "displayName": "Jay",
                "description": "ceo",
                "followersCount": 10,
                "followsCount": 2,
                "postsCount": 3,
            }
        if "getProfile" in url and "actor=pfrazee.com" in url:
            return {
                "handle": "pfrazee.com",
                "did": "did:paul",
                "displayName": "Paul",
                "description": "",
                "followersCount": 4,
                "followsCount": 1,
                "postsCount": 1,
            }
        if "getProfile" in url and "actor=switch.bsky.social" in url:
            return {"handle": "switch.bsky.social", "did": "did:switch", "displayName": "Switch", "followersCount": 1, "followsCount": 0, "postsCount": 0}
        if "getFollows" in url and "actor=jay.bsky.team" in url:
            if "cursor=" in url:
                return {"follows": [{"handle": "pfrazee.com", "did": "did:paul", "displayName": "Paul"}]}
            return {
                "follows": [{"handle": "bsky.app", "did": "did:app", "displayName": "Bluesky"}],
                "cursor": "next",
            }
        if "getFollowers" in url and "actor=jay.bsky.team" in url:
            return {"followers": [{"handle": "pfrazee.com", "did": "did:paul", "displayName": "Paul"}]}
        if "getAuthorFeed" in url and "actor=jay.bsky.team" in url:
            return {
                "feed": [
                    {
                        "post": {
                            "likeCount": 2,
                            "repostCount": 1,
                            "record": {
                                "text": "hello",
                                "createdAt": "2026-01-01T00:00:00.000Z",
                                "reply": {"parent": {"uri": "at://did/post"}},
                            },
                        }
                    }
                ]
            }
        if "getFollows" in url or "getFollowers" in url:
            return {"follows": [], "followers": []}
        if "getAuthorFeed" in url:
            return {"feed": []}
        raise AssertionError(url)

    monkeypatch.setattr(bluesky, "get_json", fake_get)
    monkeypatch.setattr(bluesky.time, "sleep", lambda seconds: slept.append(seconds))
    return calls, slept


def test_follow_edges_paginate_and_a_mutual_is_one_edge(monkeypatch):
    calls, slept = _install(monkeypatch)
    graph = bluesky.build_bluesky_layer(["jay.bsky.team", "jay.bsky.team"])
    assert graph.nodes["bluesky:jay.bsky.team"]["followersCount"] == 10
    assert graph.nodes["bluesky:jay.bsky.team"]["posts"][0]["text"] == "hello"
    assert graph.nodes["bluesky:jay.bsky.team"]["posts"][0]["likeCount"] == 2
    # jay follows pfrazee, and pfrazee follows jay, so the two lists become one mutual edge.
    assert graph.edges["bluesky:jay.bsky.team", "bluesky:pfrazee.com"]["direction"] == "mutual"
    assert graph.edges["bluesky:jay.bsky.team", "bluesky:pfrazee.com"]["weight"] == 1.0
    assert any("cursor=next" in url for url in calls)
    assert all("key=" not in url for url in calls)
    assert slept
    assert all(seconds <= 1.0 for seconds in slept)


def test_errors_return_empty_and_a_403_uses_the_fallback_host(monkeypatch, capsys):
    _install(monkeypatch)
    assert bluesky.fetch_follows("missing.bsky.social") == []
    assert "failed" in capsys.readouterr().err
    profile = bluesky.fetch_profile("switch.bsky.social")
    assert profile["handle"] == "switch.bsky.social"
    assert bluesky._host_index == 1
