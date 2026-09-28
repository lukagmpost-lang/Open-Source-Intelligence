import json

from viz.findings_page import MODULARITY_RISE, load_sweep, render_page


def test_modularity_rise_matches_the_written_ratio():
    assert abs(MODULARITY_RISE - 0.7194282555751091) < 1e-12


def test_page_carries_every_finding_and_the_sweep(tmp_path):
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            [
                {
                    "omega": 0.0,
                    "modularity": 0.5294233707,
                    "months": ["2006-05", "2007-08"],
                    "snapshots": [
                        {"communities": 12},
                        {"communities": 38},
                    ],
                    "transitions": [
                        {"earlier": "2006-05", "later": "2007-08", "mean_persistence": 0.0, "shared_members": 163}
                    ],
                },
                {
                    "omega": 0.1,
                    "modularity": 0.4672097196,
                    "months": ["2006-05", "2007-08"],
                    "snapshots": [
                        {"communities": 12},
                        {"communities": 33},
                    ],
                    "transitions": [
                        {"earlier": "2006-05", "later": "2007-08", "mean_persistence": 1.0, "shared_members": 163}
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    page = render_page(load_sweep(sweep_path))
    for anchor in (
        "finding-1",
        "finding-2",
        "finding-3",
        "finding-4",
        "finding-5",
        "finding-6",
        "finding-7",
        "finding-8",
        "finding-9",
        "finding-10",
        "multislice",
    ):
        assert f'id="{anchor}"' in page
    assert "0.3246767820" in page
    assert "0.5582584329" in page
    assert "grauenwolf" in page
    assert "DISSOLVED" in page
    assert "0.001" in page
    assert "incredible-ninja" in page
    assert "UNIVERSAL" in page
    assert "marcel" in page
    assert 'href="#networks"' in page
    assert "0.529423" in page
    assert "table.sortable" in page
    assert page.count("<script>") == 1
    assert 'class="network"' not in page


def test_page_embeds_the_six_networks(tmp_path):
    from viz.findings_page import network_drawings

    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(
        json.dumps(
            [
                {
                    "omega": 0.0,
                    "modularity": 0.5,
                    "months": ["2006-05", "2007-08"],
                    "snapshots": [{"communities": 1}, {"communities": 1}],
                    "transitions": [
                        {"earlier": "2006-05", "later": "2007-08", "mean_persistence": 0.0, "shared_members": 1}
                    ],
                },
                {
                    "omega": 0.1,
                    "modularity": 0.4,
                    "months": ["2006-05", "2007-08"],
                    "snapshots": [{"communities": 1}, {"communities": 1}],
                    "transitions": [
                        {"earlier": "2006-05", "later": "2007-08", "mean_persistence": 1.0, "shared_members": 1}
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    node = {
        "id": "ada",
        "label": "ada",
        "color": "#111111",
        "colorCommunity": "#111111",
        "colorPagerank": "#222222",
        "colorDegree": "#333333",
        "size": 20,
        "title": "ada",
    }
    panels = [{"caption": f"panel {index}", "nodes": [node], "edges": []} for index in range(6)]
    markup, scripts = network_drawings(panels)
    page = render_page(load_sweep(sweep_path), markup, scripts)
    assert page.count('class="network"') == 6
    assert 'id="search"' in page
    assert 'id="color-by"' in page
    assert "vis-network" not in page
    assert "Follow graphs, plus the SNAP friendship graph" in page
    assert "<h2>Co-participation</h2>" in page
    assert page.count("<script>") == 2
    assert "document.write" not in page
    assert "100 highest-PageRank" in page
    assert max(len(line) for line in page.splitlines()) <= 500
