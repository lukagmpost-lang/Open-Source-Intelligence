import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import compare
from osi.store import create_run, save_communities, save_metrics


def _runs(tmp_path, monkeypatch):
    monkeypatch.setenv("OSI_STORE", str(tmp_path / "store.db"))
    path = tmp_path / "store.db"
    create_run("reddit", {"layer": "reddit_user"}, "a", run_id="a", path=path)
    create_run("reddit", {"layer": "reddit_2012_user"}, "b", run_id="b", path=path)
    save_metrics("a", {"stay": 0.2, "up": 0.1, "down": 0.5, "gone": 0.4}, metric="pagerank", path=path)
    save_metrics("b", {"stay": 0.2, "up": 0.4, "down": 0.1, "new": 0.9}, metric="pagerank", path=path)
    # 0 stable, 1 split across two B communities, 2 and 3 merged into one, 4 dissolved.
    save_communities(
        "a",
        "louvain",
        {
            **{f"s{i}": 0 for i in range(10)},
            **{f"p{i}": 1 for i in range(6)},
            **{name: 2 for name in ("u", "v", "w")},
            **{name: 3 for name in ("x", "y", "z")},
            **{name: 4 for name in ("gone1", "gone2", "gone3")},
        },
        path=path,
    )
    save_communities(
        "b",
        "louvain",
        {
            **{f"s{i}": 0 for i in range(10)},
            "p0": 1,
            "p1": 1,
            "p2": 1,
            "p3": 2,
            "p4": 2,
            "p5": 2,
            **{name: 3 for name in ("u", "v", "w", "x", "y", "z")},
        },
        path=path,
    )


def test_metric_tables_split_risers_fallers_new_and_gone(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    assert compare.main(["--a", "a", "--b", "b", "--metric", "pagerank", "--top", "5"]) == 0
    text = capsys.readouterr().out
    assert "RISERS" in text and "up" in text
    assert "FALLERS" in text and "down" in text
    assert "NEW" in text and "new" in text
    assert "GONE" in text and "gone" in text
    assert "stay" not in text


def test_community_classes(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    assert compare.main(["--a", "a", "--b", "b", "--algorithm", "louvain", "--mode", "communities"]) == 0
    rows = {line.split()[0]: line.split()[-1] for line in capsys.readouterr().out.splitlines()[1:]}
    assert rows["0"] == "STABLE"
    assert rows["1"] == "SPLIT"
    assert rows["2"] == "MERGED"
    assert rows["3"] == "MERGED"
    assert rows["4"] == "DISSOLVED"


def test_missing_run_lists_what_is_stored(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    assert compare.main(["--a", "missing", "--b", "b", "--metric", "pagerank"]) == 1
    assert "a reddit" in capsys.readouterr().out
