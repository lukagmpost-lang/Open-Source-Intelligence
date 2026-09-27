import networkx as nx

from osi.temporal_communities import (
    community_persistence,
    mean_persistence,
    multislice_communities,
    multislice_modularity,
)


def _two_cliques():
    """The same two triangles in both years, with no edge between them."""
    earlier = nx.Graph()
    earlier.add_edges_from([("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f"), ("d", "f")])
    later = earlier.copy()
    return earlier, later


def test_zero_omega_does_not_keep_a_conflicting_cut():
    # The two snapshots group the same four nodes in opposite pairs.
    earlier = nx.Graph()
    earlier.add_edge("a", "b", weight=5)
    earlier.add_edge("c", "d", weight=5)
    later = nx.Graph()
    later.add_edge("a", "c", weight=5)
    later.add_edge("b", "d", weight=5)
    loose = multislice_communities([earlier, later], omega=0.0)
    tight = multislice_communities([earlier, later], omega=1.0)
    # No coupling: nobody keeps the first snapshot's community id.
    assert mean_persistence(community_persistence(loose[0], loose[1])) == 0.0
    # Coupling holds the ids even where the second snapshot's edges disagree.
    assert mean_persistence(community_persistence(tight[0], tight[1])) > 0.0


def test_high_omega_keeps_both_cliques_together():
    earlier, later = _two_cliques()
    assignments = multislice_communities([earlier, later], omega=2.0)
    rows = community_persistence(assignments[0], assignments[1])
    # Every shared member keeps the community id from the first snapshot.
    assert rows
    assert mean_persistence(rows) == 1.0
    assert {len(set(assignment.values())) for assignment in assignments} == {2}
    assert multislice_modularity([earlier, later], assignments, omega=2.0) > 0
