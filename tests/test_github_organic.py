from itertools import combinations

from osi.layers.github_organic import (
    build_github_organic,
    contributor_group,
    human_logins,
    is_bot,
    select_repos,
)


def _people(count, bot=None):
    rows = [{"login": f"u{index}", "type": "User"} for index in range(count)]
    if bot:
        rows.append(bot)
    return rows


def _fetch(repos):
    """repos maps 'owner/name' to a contributor list. Owner pages are derived from those keys."""

    def fetch(url, headers=None):
        if "/users/" in url and "/repos?" in url:
            login = url.split("/users/")[1].split("/")[0]
            owned = []
            for full_name, _contributors in repos.items():
                owner, name = full_name.split("/", 1)
                if owner != login:
                    continue
                owned.append({"name": name, "fork": False, "owner": {"login": owner}})
            return owned
        if "/contributors?" in url:
            full_name = url.split("/repos/")[1].split("/contributors")[0]
            return repos[full_name]
        raise AssertionError(url)

    return fetch


def test_shared_repos_add_weight(tmp_path):
    repos = {
        "ada/one": _people(2),
        "ada/two": _people(2),
    }
    graph = build_github_organic(("ada",), (), _fetch(repos), tmp_path)
    assert graph.edges["u0", "u1"]["weight"] == 2.0
    assert graph.nodes["u0"]["layer"] == "github_organic"


def test_over_cap_and_full_page_add_no_edges(tmp_path):
    repos = {
        "ada/wide": _people(31),
        "ada/huge": _people(100),
        "ada/pair": _people(2),
    }
    graph = build_github_organic(("ada",), (), _fetch(repos), tmp_path)
    assert graph.number_of_edges() == 1
    assert set(graph.nodes) == {"u0", "u1"}


def test_bots_are_removed_before_the_cap(tmp_path):
    bot = {"login": "dependabot[bot]", "type": "Bot"}
    repos = {"ada/ok": _people(30, bot)}
    graph = build_github_organic(("ada",), (), _fetch(repos), tmp_path)
    assert "dependabot[bot]" not in graph
    assert graph.number_of_edges() == len(list(combinations(range(30), 2)))


def test_one_human_and_a_bot_is_skipped(tmp_path):
    repos = {"ada/solo": [{"login": "ada", "type": "User"}, {"login": "news[bot]", "type": "Bot"}]}
    graph = build_github_organic(("ada",), (), _fetch(repos), tmp_path)
    assert graph.number_of_nodes() == 0


def test_cache_skips_the_second_fetch(tmp_path):
    calls = []
    inner = _fetch({"ada/pair": _people(2)})

    def fetch(url, headers=None):
        calls.append(url)
        return inner(url, headers)

    first = build_github_organic(("ada",), (), fetch, tmp_path)
    used = len(calls)
    second = build_github_organic(("ada",), (), fetch, tmp_path)
    assert len(calls) == used
    assert second.number_of_edges() == first.number_of_edges()


def test_forks_are_dropped_and_named_repos_are_kept():
    owned = [("ada", "forked", True), ("ada", "mine", False)]
    chosen = select_repos(owned, (("hadley", "ggplot2"),))
    assert ("ada", "mine") in chosen
    assert ("hadley", "ggplot2") in chosen
    assert ("ada", "forked") not in chosen


def test_bot_login_rules():
    assert is_bot("ImgBot[bot]", "User")
    assert is_bot("renovate", "User")
    assert is_bot("octo", "Bot")
    assert not is_bot("octo", "User")
    assert human_logins([{"login": "octo", "type": "User"}, {"type": "User"}]) == ["octo"]
    assert contributor_group(_people(100)) is None
    assert len(contributor_group(_people(30))) == 30
