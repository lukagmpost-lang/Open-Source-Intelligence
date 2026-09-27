import networkx as nx
import numpy as np

import osi.identity_structural as structural


def _graphs():
    github = nx.Graph()
    github.add_nodes_from(["ada", "bob"])
    reddit = nx.Graph()
    reddit.add_nodes_from(["ada", "zoe", "bob"])
    return github, reddit


def test_confirmed_structural_and_name_only_use_the_thresholds(monkeypatch):
    vectors = {
        ("github", "ada"): np.array([1.0, 0.0]),
        ("github", "bob"): np.array([0.0, 1.0]),
        ("reddit", "ada"): np.array([1.0, 0.0]),
        ("reddit", "bob"): np.array([1.0, 0.0]),
        ("reddit", "zoe"): np.array([0.95, 0.05]),
    }

    def fake_fingerprint(graph, node, metrics):
        return vectors[(metrics["tag"], node)]

    monkeypatch.setattr(structural, "compute_fingerprint", fake_fingerprint)
    github, reddit = _graphs()
    confirmed, structural_only, name_only = structural.match_by_structure(
        github,
        reddit,
        [("ada", "ada"), ("bob", "bob"), ("zoe", "bob")],
        reddit_nodes=["ada", "bob", "zoe"],
        github_metrics={"tag": "github"},
        reddit_metrics={"tag": "reddit"},
    )
    assert [(row["reddit"], row["github"]) for row in confirmed] == [("ada", "ada")]
    assert confirmed[0]["combined_score"] == 0.7 * 1.0 + 0.3
    assert [(row["reddit"], row["github"]) for row in name_only] == [("bob", "bob")]
    # zoe is not named ada, and ada is the closest GitHub node above 0.8.
    assert ("zoe", "ada") in {(row["reddit"], row["github"]) for row in structural_only}
    assert all(row["struct_sim"] > 0.8 for row in structural_only)
    assert all(not row["name_exact"] for row in structural_only)
