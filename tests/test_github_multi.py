import pytest

import suggest_identity_v2
from osi.github_graph import fetch_github_multi_ego


def _install(monkeypatch, tmp_path, profiles, followers, following):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    import osi.github_graph as github_graph

    github_graph._last_request = 0.0
    calls = []
    slept = []

    def fake_get(url, headers=None):
        calls.append(url)
        login = url.split("/users/")[1].split("/")[0].split("?")[0]
        if "/followers" in url:
            page = 2 if "page=2" in url else 1
            return followers.get((login, page), [])
        if "/following" in url:
            page = 2 if "page=2" in url else 1
            return following.get((login, page), [])
        return profiles[login]

    monkeypatch.setattr("osi.github_graph.get_json", fake_get)
    monkeypatch.setattr("osi.github_graph.time.sleep", lambda seconds: slept.append(seconds))
    return calls, slept


def test_shared_neighbor_connects_egos_and_the_store_skips_a_refetch(monkeypatch, tmp_path):
    profiles = {
        "rtomayko": {"login": "rtomayko", "followers": 1},
        "defunkt": {"login": "defunkt", "followers": 2},
    }
    followers = {
        ("rtomayko", 1): [{"login": "ada"}],
        ("defunkt", 1): [{"login": "ada"}, {"login": "rtomayko"}],
    }
    following = {
        ("rtomayko", 1): [{"login": "defunkt"}],
        ("defunkt", 1): [],
    }
    calls, slept = _install(monkeypatch, tmp_path, profiles, followers, following)
    graph = fetch_github_multi_ego(["rtomayko", "defunkt"])
    # ada sits in both neighborhoods, so the layer is not two disjoint stars.
    assert graph.degree("ada") == 2
    assert graph.edges["rtomayko", "defunkt"]["direction"] == "mutual"
    # The pause fills out the two-second gap, so a few microseconds of work make it just under 2.
    assert len(slept) == len(calls) - 1
    assert all(seconds == pytest.approx(2.0, abs=0.05) for seconds in slept)
    used = len(calls)
    again = fetch_github_multi_ego(["defunkt", "rtomayko"])
    assert len(calls) == used
    assert again.number_of_edges() == graph.number_of_edges()


def test_zero_follower_login_is_omitted_and_page_cap_is_honored(monkeypatch, tmp_path):
    profiles = {
        "akdas": {"login": "akdas", "followers": 0},
        "octocat": {"login": "octocat", "followers": 400},
    }
    page = [{"login": f"n{index}"} for index in range(100)]
    followers = {("octocat", 1): page, ("octocat", 2): [{"login": "later"}]}
    following = {("octocat", 1): page, ("octocat", 2): [{"login": "later"}]}
    calls, _slept = _install(monkeypatch, tmp_path, profiles, followers, following)
    graph = fetch_github_multi_ego(["akdas", "octocat"], max_pages=1)
    assert "akdas" not in graph
    assert "later" not in graph
    assert graph.number_of_nodes() == 101
    assert not any("/akdas/followers" in url or "page=2" in url for url in calls)


def test_github_graph_flag_is_exclusive():
    try:
        suggest_identity_v2.main(["--reddit-run", "r2008-v2", "--github-user", "rtomayko", "--github-graph", "x.graphml"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("both GitHub sources were accepted")
    try:
        suggest_identity_v2.main(["--reddit-run", "r2008-v2"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("a missing GitHub source was accepted")
