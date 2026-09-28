"""A result plus the notes a caller needs before trusting the number.

``trust_from_nmi`` labels a community partition. ``estimate_memory`` refuses
work that would pass 3000 MB. ``make_caveats`` turns those limits into sentences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResultObject:
    intent: str  # "rank_nodes", "list_communities", ...
    params: dict  # the query params
    values: dict  # numeric results (node -> score, etc.)
    method: str  # "exact" | "sampled" | "approximate"
    sample_size: int | None  # if sampled, how many
    trust: str  # "stable" | "moderate" | "unstable" | "random"
    caveats: list[str]  # human-readable limitations
    runtime_ms: int

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "params": dict(self.params),
            "values": dict(self.values),
            "method": self.method,
            "sample_size": self.sample_size,
            "trust": self.trust,
            "caveats": list(self.caveats),
            "runtime_ms": self.runtime_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResultObject:
        sample_size = d["sample_size"]
        return cls(
            intent=d["intent"],
            params=dict(d["params"]),
            values=dict(d["values"]),
            method=d["method"],
            sample_size=None if sample_size is None else int(sample_size),
            trust=d["trust"],
            caveats=list(d["caveats"]),
            runtime_ms=int(d["runtime_ms"]),
        )


def trust_from_nmi(mean_nmi: float, z: float) -> str:
    """stable if mean_nmi >= 0.9 and z >= 3
    unstable if mean_nmi < 0.75
    random if z < 2
    else moderate"""
    # Checked in that order, so a low NMI wins over a low z.
    if mean_nmi >= 0.9 and z >= 3:
        return "stable"
    if mean_nmi < 0.75:
        return "unstable"
    if z < 2:
        return "random"
    return "moderate"


def make_caveats(result: ResultObject, graph_size: int) -> list[str]:
    """Human-readable limitations based on method, sample size, and graph size."""
    caveats: list[str] = []
    if result.method == "sampled":
        if result.sample_size is None:
            caveats.append("This result is sampled, and the sample size was not recorded.")
        else:
            caveats.append(
                f"This result uses a sample of {result.sample_size} out of {graph_size} nodes, "
                "so a different sample can change the scores."
            )
    elif result.method == "approximate":
        caveats.append("This result is approximate, not an exact computation on the whole graph.")
    # A handful of nodes makes every rank swing when one edge changes.
    if graph_size < 10:
        caveats.append(
            f"The graph has only {graph_size} nodes, so one added or removed edge can change the result."
        )
    return caveats


# Exact betweenness stores a dependency score per node per edge. The 100 is that working set in bytes, scaled to MB.
_BETWEENNESS_EXACT_BYTES = 100
# A 500-source sample touches each edge a constant number of times.
_BETWEENNESS_SAMPLE_K = 500
_SAFE_MB = 3000
_CPM_NODE_CAP = 3000


def estimate_memory(nodes: int, edges: int, algorithm: str) -> dict:
    """Return dict with estimated_mb, safe (bool), and recommended_downgrade.

    Rough rules:
    - betweenness (exact): 100 * nodes * edges / 1e6 MB
    - betweenness (sampled, k=500): 500 * edges / 1e6 MB
    - louvain/leiden: edges / 1e6 * 2 MB
    - cpm: skip above 3000 nodes (exponential in degree)
    - multislice (S slices): S * edges / 1e6 * 3 MB

    safe=False if estimated_mb > 3000.
    """
    name, slices = _algorithm_parts(algorithm)
    if name == "betweenness":
        estimated = _BETWEENNESS_EXACT_BYTES * nodes * edges / 1e6
        downgrade = "betweenness_sampled"
    elif name == "betweenness_sampled":
        estimated = _BETWEENNESS_SAMPLE_K * edges / 1e6
        downgrade = "louvain"
    elif name in {"louvain", "leiden"}:
        estimated = edges / 1e6 * 2
        downgrade = None
    elif name == "cpm":
        if nodes > _CPM_NODE_CAP:
            return {"estimated_mb": None, "safe": False, "recommended_downgrade": "louvain"}
        estimated = edges / 1e6 * 2
        downgrade = "louvain"
    elif name == "multislice":
        estimated = slices * edges / 1e6 * 3
        downgrade = "louvain"
    else:
        raise ValueError(f"unknown algorithm {algorithm}")
    safe = estimated <= _SAFE_MB
    return {
        "estimated_mb": estimated,
        "safe": safe,
        "recommended_downgrade": None if safe else downgrade,
    }


def _algorithm_parts(algorithm: str) -> tuple[str, int]:
    """Split ``multislice:6`` into the name and the slice count. Other names use 1."""
    text = algorithm.strip().lower()
    if text.startswith("multislice:"):
        return "multislice", int(text.split(":", 1)[1])
    return text, 1
