"""Read an edge list or a GraphML file into a NetworkX graph.

CSV and TSV use the csv module so quoted fields stay intact. JSON accepts a
NetworkX node-link document or a list of ``{source, target, weight}`` objects.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import networkx as nx


def _print_summary(graph: nx.Graph) -> None:
    # A directed graph is "connected" when you can travel either way along edges.
    if graph.is_directed():
        components = nx.number_weakly_connected_components(graph)
    else:
        components = nx.number_connected_components(graph)
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    node_word = "node" if nodes == 1 else "nodes"
    edge_word = "edge" if edges == 1 else "edges"
    component_word = "connected component" if components == 1 else "connected components"
    print(f"{nodes} {node_word}, {edges} {edge_word}, {components} {component_word}")


def _as_weight(raw: object) -> float:
    if raw is None or raw == "":
        return 1.0
    return float(raw)


def _add_record(graph: nx.Graph, source: object, target: object, raw_weight: object, where: str) -> None:
    if source is None or target is None or str(source).strip() == "" or str(target).strip() == "":
        print(f"warning: skipped {where}: missing source or target", file=sys.stderr)
        return
    try:
        weight = _as_weight(raw_weight)
    except (TypeError, ValueError):
        print(f"warning: skipped {where}: weight {raw_weight!r} is not a number", file=sys.stderr)
        return
    graph.add_edge(source, target, weight=weight)


def _from_table(path: Path, source_col: str, target_col: str, weight_col: str | None, directed: bool) -> nx.Graph:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    # newline="" is what the csv module needs so quoted newlines are not split twice.
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = list(reader.fieldnames or [])
        if source_col not in fields or target_col not in fields:
            raise ValueError(f"{path} needs columns {source_col!r} and {target_col!r}")
        # A weight column is optional. A blank cell is the same as leaving the column out.
        column = weight_col if weight_col and weight_col in fields else ("weight" if "weight" in fields else None)
        if weight_col and weight_col not in fields:
            print(f"warning: weight column {weight_col!r} is missing; using 1.0", file=sys.stderr)
        for row in reader:
            _add_record(graph, row.get(source_col), row.get(target_col), row.get(column) if column else None, f"row {reader.line_num}")
    return graph


def _from_records(rows: list, source_col: str, target_col: str, weight_col: str | None, directed: bool) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    key = weight_col or "weight"
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            print(f"warning: skipped item {index}: expected an object", file=sys.stderr)
            continue
        raw_weight = row.get(key) if key in row else None
        _add_record(graph, row.get(source_col), row.get(target_col), raw_weight, f"item {index}")
    return graph


def _from_node_link(payload: dict) -> nx.Graph:
    # NetworkX 3.x automatically handles both "links" and "edges" keys
    return nx.node_link_graph(payload)


def load_edge_list(
    path,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = None,
    directed: bool = False,
) -> nx.Graph:
    """Load a CSV, TSV, or JSON edge list into a NetworkX graph.

    Auto-detect format from the file extension.
    For CSV/TSV: read with the csv module. Handle quoted fields, skip
    malformed rows with a warning, treat a missing weight as 1.0.
    For JSON: accept both NetworkX node-link format and a plain list
    of {source, target, weight} objects.
    Print: node count, edge count, connected components.
    Return the graph.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        graph = _from_table(file_path, source_col, target_col, weight_col, directed)
    elif suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            graph = _from_records(payload, source_col, target_col, weight_col, directed)
        elif isinstance(payload, dict) and "nodes" in payload:
            # The document already says whether it is directed. The flag is for tables and lists.
            graph = _from_node_link(payload)
        else:
            raise ValueError("JSON must be a node-link object or a list of {source, target, weight}")
    else:
        raise ValueError(f"unsupported edge-list extension {suffix or '(none)'}; use .csv, .tsv, or .json")
    _print_summary(graph)
    return graph


def load_graphml(path) -> nx.Graph:
    """Load a GraphML file into NetworkX via nx.read_graphml.

    Preserve node and edge attributes. Prints the same node, edge, and
    component counts as load_edge_list.
    """
    graph = nx.read_graphml(path)
    _print_summary(graph)
    return graph
