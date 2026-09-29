from osi.result import ResultObject, estimate_memory, make_caveats, trust_from_nmi


def test_community_values_names_the_largest_group():
    values = ResultObject.community_values({2: 3, 0: 10, 1: 3}, modularity=0.42)
    assert values == {
        "n_communities": 3,
        "modularity": 0.42,
        "sizes": [10, 3, 3],
        "largest_community": 0,
        "largest_size": 10,
    }


def test_community_values_breaks_size_ties_by_id():
    values = ResultObject.community_values({1: 4, 0: 4}, modularity=None)
    assert values["n_communities"] == 2
    assert values["sizes"] == [4, 4]
    assert values["largest_community"] == 0
    assert values["largest_size"] == 4
    assert values["modularity"] is None


def test_result_object_round_trips():
    original = ResultObject(
        intent="rank_nodes",
        params={"metric": "pagerank", "top": 5},
        values={"alice": 0.4, "bob": 0.3},
        method="exact",
        sample_size=None,
        trust="stable",
        caveats=["tiny graph"],
        runtime_ms=12,
    )
    payload = original.to_dict()
    restored = ResultObject.from_dict(payload)
    assert restored == original
    payload["caveats"].append("changed")
    assert original.caveats == ["tiny graph"]


def test_trust_from_nmi_boundaries():
    assert trust_from_nmi(0.9, 3) == "stable"
    assert trust_from_nmi(1.0, 4) == "stable"
    # Just under either stable cutoff, and not in the unstable or random bands.
    assert trust_from_nmi(0.9, 2.999) == "moderate"
    assert trust_from_nmi(0.899, 3) == "moderate"
    # 0.75 is still above the unstable cutoff.
    assert trust_from_nmi(0.75, 3) == "moderate"
    assert trust_from_nmi(0.749, 5) == "unstable"
    # A low NMI is unstable even when z is also in the random band.
    assert trust_from_nmi(0.74, 1) == "unstable"
    # z of 2 is not random. Just under 2 is, including when NMI would otherwise look strong.
    assert trust_from_nmi(0.8, 2) == "moderate"
    assert trust_from_nmi(0.8, 1.999) == "random"
    assert trust_from_nmi(0.9, 1.5) == "random"
    assert trust_from_nmi(0.8, 2.5) == "moderate"


def test_sampled_result_has_a_caveat():
    result = ResultObject(
        intent="rank_nodes",
        params={"metric": "betweenness"},
        values={"a": 0.2},
        method="sampled",
        sample_size=500,
        trust="moderate",
        caveats=[],
        runtime_ms=40,
    )
    caveats = make_caveats(result, graph_size=50_000)
    assert len(caveats) >= 1
    assert any("500" in caveat for caveat in caveats)


def test_exact_betweenness_on_50k_nodes_is_unsafe():
    # 100 * 50000 * 50000 / 1e6 = 250_000 MB, well past the 3000 MB line.
    estimate = estimate_memory(50_000, 50_000, "betweenness")
    assert estimate["safe"] is False
    assert estimate["estimated_mb"] > 3000
    assert estimate["recommended_downgrade"] == "betweenness_sampled"
