import pytest

from osi.executors import baseline_compare


def test_reddit_2012_against_snap_returns_ratios_and_findings():
    result = baseline_compare("r2012-v2", baseline="snap_facebook")
    values = result.values
    assert values["baseline_name"] == "snap_facebook"
    assert values["ratios"]
    assert values["this_graph"]["components"]
    assert values["baseline_metrics"]["components"] == 1
    assert values["findings"]
    text = " ".join(values["findings"])
    assert any(
        name in text
        for name in (
            "nodes",
            "edges",
            "modularity",
            "avg_degree",
            "max_degree",
            "components",
            "clustering",
            "assortativity",
        )
    )


def test_an_unknown_baseline_names_the_problem():
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_compare("r2012-v2", baseline="not_a_network")
