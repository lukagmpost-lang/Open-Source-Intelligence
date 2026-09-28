from osi.layers.bluesky_organic import (
    authors_in_thread,
    build_bluesky_organic,
    is_bot,
    thread_root,
)


def _feed(roots):
    rows = []
    for uri in roots:
        rows.append({"post": {"uri": uri, "author": {"handle": "jay.bsky.team"}, "record": {}}})
    return {"feed": rows}


def _thread(handles):
    return {
        "thread": {
            "post": {"author": {"handle": handles[0]}},
            "replies": [{"post": {"author": {"handle": handle}}} for handle in handles[1:]],
        }
    }


def test_shared_threads_add_weight(tmp_path):
    def fetch(url):
        if "getAuthorFeed" in url:
            return _feed(["at://one", "at://two"])
        if "uri=at%3A%2F%2Fone" in url:
            return _thread(["jay.bsky.team", "pfrazee.com"])
        if "uri=at%3A%2F%2Ftwo" in url:
            return _thread(["jay.bsky.team", "pfrazee.com", "emilyliu.me"])
        raise AssertionError(url)

    graph = build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    assert graph.edges["jay.bsky.team", "pfrazee.com"]["weight"] == 2.0
    assert graph.edges["pfrazee.com", "emilyliu.me"]["weight"] == 1.0
    assert graph.nodes["jay.bsky.team"]["layer"] == "bluesky_organic"


def test_over_cap_thread_is_skipped(tmp_path):
    def fetch(url):
        if "getAuthorFeed" in url:
            return _feed(["at://wide"])
        return _thread([f"user{index}.bsky.social" for index in range(31)])

    graph = build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    assert graph.number_of_nodes() == 0


def test_bot_is_removed_before_the_cap(tmp_path):
    handles = [f"user{index}.bsky.social" for index in range(30)] + ["news[bot]"]

    def fetch(url):
        if "getAuthorFeed" in url:
            return _feed(["at://ok"])
        return _thread(handles)

    graph = build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    assert "news[bot]" not in graph
    assert graph.number_of_nodes() == 30


def test_parent_and_replies_are_both_authors():
    node = {
        "post": {"author": {"handle": "child.bsky.social"}},
        "parent": {"post": {"author": {"handle": "root.bsky.social"}}},
        "replies": [{"post": {"author": {"handle": "other.bsky.social"}}}, {"post": {"author": {"handle": "news[bot]"}}}],
    }
    assert authors_in_thread(node) == {"child.bsky.social", "root.bsky.social", "other.bsky.social"}


def test_reply_uses_the_root_uri(tmp_path):
    seen = []

    def fetch(url):
        seen.append(url)
        if "getAuthorFeed" in url:
            return {
                "feed": [
                    {
                        "post": {
                            "uri": "at://did/app.bsky.feed.post/reply",
                            "record": {"reply": {"root": {"uri": "at://did/app.bsky.feed.post/root"}}},
                        }
                    }
                ]
            }
        return _thread(["jay.bsky.team", "pfrazee.com"])

    build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    assert any("post%2Froot" in url for url in seen)
    assert not any("post%2Freply" in url and "getPostThread" in url for url in seen)


def test_cache_skips_the_second_fetch(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        if "getAuthorFeed" in url:
            return _feed(["at://one"])
        return _thread(["jay.bsky.team", "pfrazee.com"])

    build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    used = len(calls)
    build_bluesky_organic(("jay.bsky.team",), fetch, tmp_path)
    assert len(calls) == used


def test_thread_root_and_bot_helper():
    assert thread_root({"uri": "at://post", "record": {}}) == "at://post"
    assert (
        thread_root({"uri": "at://reply", "record": {"reply": {"root": {"uri": "at://root"}}}})
        == "at://root"
    )
    assert is_bot("handle.invalid")
    assert is_bot("news[bot]")
    assert not is_bot("jay.bsky.team")
