import json

import networkx as nx

from osi.identity import annotate_coverage, apply_identity_map, fetch_github_identities, load_identity_map, merge_identity_layers


def test_identity_merges_one_person_across_github_and_reddit(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps({"person_a": {"github": "ada", "reddit": "ada_r", "steam": "missing"}}),
        encoding="utf-8",
    )
    identity = load_identity_map(path)
    github = nx.Graph()
    github.add_node("github:ada", color="blue")
    github.add_edge("github:ada", "github:grace", weight=2.0)
    reddit = nx.Graph()
    reddit.add_node("reddit:ada_r", color="orange")
    reddit.add_edge("reddit:ada_r", "reddit:linus", weight=1.0)
    # steam is named in the map and absent from the layers. That must not abort the merge.
    merged = merge_identity_layers({"github": github, "reddit": reddit, "steam": None}, identity)
    assert merged.nodes["person_a|github"]["color"] == "blue"
    assert merged.nodes["person_a|github"]["person"] == "person_a"
    assert merged.nodes["person_a|reddit"]["color"] == "orange"
    assert merged.edges["person_a|github", "github:grace"]["kind"] == "intralayer"
    assert merged.edges["person_a|github", "github:grace"]["layer"] == "github"
    assert merged.edges["person_a|reddit", "reddit:linus"]["kind"] == "intralayer"
    assert merged.edges["person_a|reddit", "reddit:linus"]["layer"] == "reddit"
    assert merged.edges["person_a|github", "person_a|reddit"]["kind"] == "interlayer"
    assert "github:ada" not in merged and "reddit:ada_r" not in merged
    rewritten = apply_identity_map(github, "github", identity)
    assert "person_a|github" in rewritten and rewritten.nodes["person_a|github"]["color"] == "blue"
    assert "github:grace" in rewritten


def test_coverage_marks_a_person_github_only_when_reddit_is_missing():
    identity = {
        "linus_torvalds": {"github": "torvalds", "reddit": "not-there"},
        "ada": {"github": "ada", "reddit": "ada_r"},
    }
    github = nx.Graph()
    github.add_node("torvalds")
    github.add_node("ada")
    github.add_node("grace")
    reddit = nx.Graph()
    reddit.add_node("ada_r")
    merged = merge_identity_layers({"github": github, "reddit": reddit}, identity)
    listed = annotate_coverage(merged)
    assert listed["both"] == ["ada"]
    assert listed["github-only"] == ["linus_torvalds"]
    assert merged.nodes["linus_torvalds|github"]["coverage"] == "github-only"
    assert merged.nodes["ada|github"]["coverage"] == "both"
    assert merged.nodes["ada|reddit"]["coverage"] == "both"
    assert "linus_torvalds|reddit" not in merged
    # A follower who is not in the identity map stays github-only.
    assert merged.nodes["github:grace"]["coverage"] == "github-only"


def test_missing_github_account_is_skipped(monkeypatch, capsys):
    def fake_fetch(username: str) -> nx.Graph:
        if username == "missing":
            raise RuntimeError("upstream HTTP 404")
        graph = nx.Graph()
        graph.add_node(username)
        return graph

    monkeypatch.setattr("osi.identity.fetch_github_graph", fake_fetch)
    graph = fetch_github_identities(
        {"linus_torvalds": {"github": "missing"}, "ada": {"github": "ada", "reddit": "ada_r"}}
    )
    assert list(graph.nodes) == ["ada"]
    assert "github handle missing for linus_torvalds skipped" in capsys.readouterr().err
