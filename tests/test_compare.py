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
            # Enough later communities that a perfect containment clears lift 50.
            **{f"f{i}": 100 + i for i in range(60)},
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


def test_lift_is_containment_times_later_communities():
    assert compare._lift(0.5, 1141) == 0.5 * 1141
    assert compare._classify(60, 1, False) == "STABLE"
    assert compare._classify(60, 2, True) == "SPLIT"
    assert compare._classify(20, 1, True) == "MERGED"
    assert compare._classify(5, 0, False) == "DISSOLVED"
    assert compare._classify(20, 1, False) == "unclassified"


def test_community_classes(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    assert compare.main(["--a", "a", "--b", "b", "--algorithm", "louvain", "--mode", "communities"]) == 0
    lines = capsys.readouterr().out.splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("a_community"))
    forward = []
    for line in lines[header + 1 :]:
        if line == "FED":
            break
        if line.strip():
            forward.append(line)
    rows = {line.split()[0]: line.split()[-1] for line in forward}
    assert rows["0"] == "STABLE"
    assert rows["1"] == "SPLIT"
    assert rows["2"] == "MERGED"
    assert rows["3"] == "MERGED"
    assert rows["4"] == "DISSOLVED"
    fed = "\n".join(lines[lines.index("FED") + 1 :])
    assert "2:64.00" in fed and "3:64.00" in fed


def test_percentile_mode_uses_stored_ranks(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    path = tmp_path / "store.db"
    save_metrics("a", {"x": 1.0, "y": 3.0, "z": 2.0}, metric="degree", path=path)
    save_metrics("b", {"x": 4.0, "y": 1.0, "z": 2.0}, metric="degree", path=path)
    assert compare.main(["--a", "a", "--b", "b", "--metric", "pagerank", "--mode", "percentile"]) == 0
    text = capsys.readouterr().out
    risers, rest = text.split("FALLERS", 1)
    fallers, rest = rest.split("ONLY_A", 1)
    only_a, rest = rest.split("ONLY_B", 1)
    only_b, top = rest.split("TOP100", 1)
    assert "up" in risers and "-50.0000%" in risers
    assert "stay" not in risers and "down" not in risers
    assert "down" in fallers and "+75.0000%" in fallers
    assert "stay" not in fallers
    assert only_a.startswith(" 1") and "gone" in only_a and "50.0000%" in only_a
    assert only_b.startswith(" 1") and "new" in only_b and "25.0000%" in only_b
    assert "stay" in top and "up" in top and "down" in top
    assert "gone" not in top and "new" not in top
    assert compare.main(["--a", "a", "--b", "b", "--metric", "betweenness", "--mode", "percentile"]) == 1
    assert "has no betweenness" in capsys.readouterr().out
    assert compare.main(["--a", "a", "--b", "b", "--metric", "degree", "--mode", "percentile"]) == 0
    degree = capsys.readouterr().out
    assert "x" in degree.split("FALLERS", 1)[0]
    assert "y" in degree.split("FALLERS", 1)[1].split("ONLY_A", 1)[0]


def test_missing_run_lists_what_is_stored(tmp_path, monkeypatch, capsys):
    _runs(tmp_path, monkeypatch)
    assert compare.main(["--a", "missing", "--b", "b", "--metric", "pagerank"]) == 1
    assert "a reddit" in capsys.readouterr().out
