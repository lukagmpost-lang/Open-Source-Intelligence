import pytest

from conftest import has_run
from osi.executors import baseline_compare


@pytest.mark.skipif(not has_run("r2012-v2"), reason="Reddit 2012 run not in the store")
def test_reddit_2012_against_snap_returns_ratios_and_findings():
    result = baseline_compare("r2012-v2", baseline="snap_facebook")
    values = result.values
    assert values["baseline_name"] == "snap_facebook"
    assert values["ratios"]
    assert values["this_graph"]["components"]
    assert values["baseline_metrics"]["components"] == 1
    assert values["findings"]
    text = " ".join(values["findings"]).lower()
    assert "components" in values["ratios"]
    assert all("means" in line or "because" in line for line in values["findings"])
    for word in ("modularity", "assortativity", "components", "density"):
        assert word not in text


def test_an_unknown_baseline_names_the_problem():
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_compare("r2012-v2", baseline="not_a_network")
