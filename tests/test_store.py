import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from osi.store import (
    create_run,
    delete_run,
    get_run,
    list_runs,
    load_communities,
    load_graph,
    load_metrics,
    save_communities,
    save_graph,
    save_metrics,
)


def _weighted_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge(7, "a", weight=1.0)
    return graph


def test_graph_roundtrip_keeps_weights_and_node_types(tmp_path):
    path = tmp_path / "store.db"
    run_id = create_run("reddit", {"layer": "reddit_user"}, "r2008-v1", run_id="r2008-v1", path=path)
    save_graph(run_id, "reddit_user", _weighted_graph(), path=path)
    loaded = load_graph(run_id, "reddit_user", path=path)
    assert loaded is not None
    assert set(loaded.nodes) == {"a", "b", 7}
    assert loaded["a"]["b"]["weight"] == 2.0
    assert loaded[7]["a"]["weight"] == 1.0
    assert load_graph(run_id, "missing", path=path) is None


def test_metrics_and_communities_roundtrip(tmp_path):
    path = tmp_path / "store.db"
    run_id = create_run("github", {"layer": "github"}, notes="tiny", path=path)
    save_metrics(run_id, {"b": 0.2, "a": 0.5, 3: 0.2}, metric="pagerank", path=path)
    ranks = load_metrics(run_id, "pagerank", path=path)
    # Equal scores break ties by node text, matching the analysis sort.
    assert list(ranks) == ["a", 3, "b"]
    assert ranks["a"] == 0.5
    assert ranks[3] == 0.2
    save_communities(run_id, "louvain", {"a": 0, "b": 1, 3: 0}, path=path)
    communities = load_communities(run_id, "louvain", path=path)
    assert communities == {"a": 0, "b": 1, 3: 0}
    assert load_communities(run_id, "leiden", path=path) == {}


def test_list_get_and_delete_run(tmp_path):
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "reddit_user"}, "first", run_id="first", path=path)
    create_run("snap_facebook", {"layer": "snap_facebook"}, "second", run_id="second", path=path)
    listed = list_runs(path=path)
    assert [item[0] for item in listed] == ["first", "second"]
    meta = get_run("first", path=path)
    assert meta is not None
    assert meta["source"] == "reddit"
    assert meta["notes"] == "first"
    assert meta["config"]["layer"] == "reddit_user"
    save_graph("first", "reddit_user", _weighted_graph(), path=path)
    delete_run("first", path=path)
    assert get_run("first", path=path) is None
    assert load_graph("first", "reddit_user", path=path) is None
    assert [item[0] for item in list_runs(path=path)] == ["second"]
