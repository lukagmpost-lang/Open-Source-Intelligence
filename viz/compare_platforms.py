"""One HTML page with the six stored platform graphs.

Run:
    python3 viz/compare_platforms.py

The store is read only. Nothing is saved back into a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

from osi.analysis import _partition_record
from osi.store import get_run, load_communities, load_graph, load_metrics, load_result
from viz.interactive import _community_color, populate_pyvis

# Top row is the follow graphs plus SNAP friendship. Bottom row is co-participation.
# That is the 2 by 3 the page draws. SNAP is friendship, so its caption says so.
PANELS = (
    {"run_id": "github-v1", "name": "GitHub follows", "family": "follow", "halving": 0.01},
    {"run_id": "bluesky-v1", "name": "Bluesky follows", "family": "follow", "halving": 0.01},
    {"run_id": "snap-v1", "name": "SNAP Facebook", "family": "friendship", "halving": 0.30},
    {"run_id": "r2008-v2", "name": "Reddit 2008", "family": "co-participation", "halving": 0.30},
    {"run_id": "github-organic-v1", "name": "GitHub co-contribution", "family": "co-participation", "halving": 0.02},
    {"run_id": "bluesky-organic-v1", "name": "Bluesky replies", "family": "co-participation", "halving": 0.10},
)
# First drawing uses this many nodes. A page over the byte cap is redrawn at the lower cap.
NODE_CAP = 800
NODE_CAP_FALLBACK = 500
# One self-contained file. Six copies of vis.js would pass this on library size alone.
MAX_BYTES = 20 * 1024 * 1024
# vis stabilization default is 1000 iterations. Six panels at that count freeze the tab.
STABILIZATION_ITERATIONS = 40
OUTPUT = ROOT / "compare_platforms.html"


def caption_line(name: str, nodes: int, edges: int, modularity: float, clustering: float, halving: float) -> str:
    """The one-line caption above a panel. Counts are the full stored graph."""
    return (
        f"{name} | {nodes} nodes | {edges} edges | "
        f"modularity {modularity:.6f} | clustering {clustering:.6f} | halving {halving:.2f}"
    )


def cap_by_pagerank(graph: nx.Graph, scores: dict, cap: int) -> nx.Graph:
    """Induced subgraph of the highest-PageRank nodes. Ties break by name."""
    ranked = sorted(graph.nodes, key=lambda node: (-float(scores.get(node, 0.0)), str(node)))
    # A graph smaller than the cap is drawn whole. The caption still reports the full size.
    keep = ranked[:cap]
    return graph.subgraph(keep).copy()


def display_label(node) -> str:
    """Label a node with the handle a person would type into the search box."""
    text = str(node)
    # The follow layer stores bluesky:handle. The handle is the searchable name.
    if text.startswith("bluesky:"):
        return text.split(":", 1)[1]
    return text


def _by_text(mapping: dict) -> dict[str, object]:
    return {str(key): value for key, value in mapping.items()}


def align_scores(graph: nx.Graph, scores: dict) -> dict:
    """Map stored scores onto graph nodes when the store and the graph disagree on int versus str."""
    by_text = _by_text(scores)
    aligned = {}
    for node in graph.nodes:
        if node in scores:
            aligned[node] = float(scores[node])
        elif str(node) in by_text:
            aligned[node] = float(by_text[str(node)])
        else:
            aligned[node] = 0.0
    return aligned


def align_communities(graph: nx.Graph, communities: dict) -> dict:
    """Same alignment for Louvain ids. A node missing from the store stays in community 0."""
    by_text = _by_text(communities)
    aligned = {}
    for node in graph.nodes:
        if node in communities:
            aligned[node] = int(communities[node])
        elif str(node) in by_text:
            aligned[node] = int(by_text[str(node)])
        else:
            aligned[node] = 0
    return aligned


def _ramp(values: dict) -> dict:
    """Viridis hex color per node. The scale is the visible panel, so the range is used."""
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    colors = {}
    for node, value in values.items():
        # One value, or a flat panel, sits in the middle of the colormap.
        if high <= low:
            fraction = 0.5
        else:
            fraction = (float(value) - low) / (high - low)
        red, green, blue, _alpha = plt.colormaps["viridis"](fraction)
        colors[node] = f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"
    return colors


def _panel_size(score: float, low: float, high: float) -> float:
    # 6 to 28 fits a grid cell. The single-graph writer still uses PageRank times 20000.
    if high <= low:
        return 14.0
    return 6.0 + 22.0 * (float(score) - low) / (high - low)


def panel_records(graph: nx.Graph, scores: dict, communities: dict, degrees: dict) -> tuple[list, list]:
    """PyVis node and edge records for one capped graph."""
    low = min(scores.values()) if scores else 0.0
    high = max(scores.values()) if scores else 0.0
    pagerank_colors = _ramp(scores)
    degree_colors = _ramp(degrees)

    def size_for(node, score):
        return _panel_size(score, low, high)

    def label_for(node):
        return display_label(node)

    def title_for(node, score, community):
        return (
            f"user: {display_label(node)} | community: {community} | "
            f"pagerank: {score:.6f} | degree: {int(degrees.get(node, 0))}"
        )

    def fields_for(node, score, community):
        # The live color starts as the community. The other two wait for the dropdown.
        return {
            "colorCommunity": _community_color(community),
            "colorPagerank": pagerank_colors.get(node, "#888888"),
            "colorDegree": degree_colors.get(node, "#888888"),
        }

    net = Network(directed=False)
    populate_pyvis(
        net,
        graph,
        scores,
        communities,
        size_for=size_for,
        label_for=label_for,
        title_for=title_for,
        fields_for=fields_for,
    )
    return net.nodes, net.edges


def _vis_source() -> str:
    """The same vis-network file PyVis inlines. One copy serves all six panels."""
    import pyvis

    path = Path(pyvis.__file__).resolve().parent / "templates" / "lib" / "vis-9.1.2" / "vis-network.min.js"
    return path.read_text(encoding="utf-8")


def _break_long_lines(source: str, limit: int = 2000) -> str:
    """Break after a comma or semicolon so no line is millions of characters long.

    A single multi-megabyte line is why a downloaded HTML file fails to open in
    an editor or in some browsers. Breaks stay outside strings and comments.
    """
    out: list[str] = []
    line_len = 0
    in_string = ""
    escape = False
    i = 0
    length = len(source)
    while i < length:
        ch = source[i]
        if in_string:
            out.append(ch)
            line_len += 1
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_string:
                in_string = ""
            i += 1
            continue
        if ch == "/" and i + 1 < length and source[i + 1] == "/":
            while i < length and source[i] != "\n":
                out.append(source[i])
                i += 1
            line_len = 0
            continue
        if ch == "/" and i + 1 < length and source[i + 1] == "*":
            out.append(ch)
            out.append(source[i + 1])
            i += 2
            while i < length and not (source[i - 1] == "*" and source[i] == "/"):
                out.append(source[i])
                if source[i] == "\n":
                    line_len = 0
                i += 1
            if i < length:
                out.append(source[i])
                i += 1
            continue
        if ch in "\"'`":
            in_string = ch
            out.append(ch)
            line_len += 1
            i += 1
            continue
        out.append(ch)
        if ch == "\n":
            line_len = 0
        else:
            line_len += 1
        if ch in ",;" and line_len >= limit:
            out.append("\n")
            line_len = 0
        i += 1
    return "".join(out)


def _script_json(payload) -> str:
    # A label containing "<" must not close the script tag.
    # Newlines keep the file openable after download. One line of graph data is several MB.
    compact = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    return _break_long_lines(compact)


def render_html(panels: list[dict], vis_js: str) -> str:
    """Full page. ``panels`` already carry caption, nodes, and edges."""
    cards = []
    for index, panel in enumerate(panels):
        # A heading before the first card of each row names the edge family.
        if index == 0:
            cards.append("<h2>Follow graphs, plus the SNAP friendship graph</h2>")
        if index == 3:
            cards.append("<h2>Co-participation</h2>")
        element = f"net-{index}"
        cards.append(
            "<section class=\"card\">"
            f"<p class=\"caption\">{panel['caption']}</p>"
            f"<div id=\"{element}\" class=\"network\"></div>"
            "</section>"
        )
    data = [
        {
            "element": f"net-{index}",
            "nodes": panel["nodes"],
            "edges": panel["edges"],
        }
        for index, panel in enumerate(panels)
    ]
    table_rows = []
    for panel in panels:
        metrics = panel["metrics"]
        table_rows.append(
            "<tr>"
            f"<td data-value=\"{panel['name']}\">{panel['name']}</td>"
            f"<td data-value=\"{metrics['clustering']}\">{metrics['clustering']:.6f}</td>"
            f"<td data-value=\"{metrics['assortativity']}\">{metrics['assortativity']:.6f}</td>"
            f"<td data-value=\"{metrics['modularity']}\">{metrics['modularity']:.6f}</td>"
            f"<td data-value=\"{metrics['halving']}\">{metrics['halving']:.2f}</td>"
            "</tr>"
        )
    # Physics stays on, but stabilization stops after a short run so six layouts can share the tab.
    options = {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.25,
                "springLength": 90,
                "springConstant": 0.04,
                "damping": 0.7,
                "avoidOverlap": 0.15,
            },
            "stabilization": {"enabled": True, "iterations": STABILIZATION_ITERATIONS, "fit": True},
        },
        "interaction": {"hover": True},
        "edges": {"smooth": False, "color": "#d6d3d1", "width": 0.4},
        "nodes": {"font": {"size": 11, "color": "#44403c"}},
    }
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Edge Semantics Determines Network Structure</title>
<style>
body {{ margin: 0; font-family: "Iowan Old Style", Palatino, Georgia, serif; color: #1c1917; background: #fafaf9; }}
header, .toolbar {{ padding: 16px 24px; }}
h1 {{ font-size: 28px; margin: 0 0 12px; }}
h2 {{ font-size: 18px; margin: 8px 0; grid-column: 1 / -1; }}
table {{ border-collapse: collapse; background: #fff; }}
th, td {{ border: 1px solid #e7e5e4; padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
th {{ cursor: pointer; background: #f5f5f4; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.toolbar {{ display: flex; gap: 16px; align-items: center; position: sticky; top: 0; background: #fafaf9; z-index: 2; }}
input, select, button {{ font: 16px system-ui, sans-serif; padding: 6px 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; padding: 0 24px 24px; }}
.card {{ background: #fff; border: 1px solid #e7e5e4; min-width: 0; }}
.caption {{ margin: 0; padding: 8px 10px; font-size: 13px; line-height: 1.4; }}
.network {{ height: 420px; }}
</style>
</head>
<body>
<header>
<h1>Edge Semantics Determines Network Structure</h1>
<table id="metrics">
<thead>
<tr>
<th data-type="text">platform</th>
<th data-type="num">clustering</th>
<th data-type="num">assortativity</th>
<th data-type="num">modularity</th>
<th data-type="num">halving point</th>
</tr>
</thead>
<tbody>
{"".join(table_rows)}
</tbody>
</table>
</header>
<div class="toolbar">
<input id="search" type="search" placeholder="Highlight a handle in all six graphs" autocomplete="off">
<button id="physics-toggle" type="button">Freeze physics</button>
<label>color by
<select id="color-by">
<option value="community">community</option>
<option value="pagerank">pagerank</option>
<option value="degree">degree</option>
</select>
</label>
</div>
<div class="grid">
{"".join(cards)}
</div>
<script>
{_vis_script(vis_js)}
</script>
<script>
const PANELS = {_script_json(data)};
const OPTIONS = {_script_json(options)};
const networks = [];
const COLOR_FIELD = {{ community: "colorCommunity", pagerank: "colorPagerank", degree: "colorDegree" }};

function draw(panel) {{
  const container = document.getElementById(panel.element);
  const data = {{
    nodes: new vis.DataSet(panel.nodes),
    edges: new vis.DataSet(panel.edges),
  }};
  const network = new vis.Network(container, data, OPTIONS);
  network.dataset = data.nodes;
  networks.push(network);
}}

// Six Barnes-Hut starts at once stall the first paint. Stagger them by a frame.
PANELS.forEach((panel, index) => {{
  setTimeout(() => draw(panel), index * 40);
}});

function matchingIds(network, query) {{
  const nodes = network.dataset.get();
  const folded = query.toLowerCase();
  const hits = nodes.filter((node) => {{
    const label = String(node.label).toLowerCase();
    const id = String(node.id).toLowerCase();
    return label.includes(folded) || id.includes(folded);
  }});
  // An exact handle wins over every node that merely contains those letters.
  const exact = hits.filter((node) => {{
    const label = String(node.label).toLowerCase();
    const id = String(node.id).toLowerCase();
    return label === folded || id === folded || id.endsWith(":" + folded);
  }});
  return (exact.length ? exact : hits).map((node) => node.id);
}}

document.getElementById("search").addEventListener("input", (event) => {{
  const query = event.target.value.trim();
  networks.forEach((network) => {{
    if (!query) {{
      network.unselectAll();
      return;
    }}
    const ids = matchingIds(network, query);
    network.selectNodes(ids);
    if (ids.length) {{
      network.fit({{ nodes: ids, animation: false }});
    }}
  }});
}});

document.getElementById("physics-toggle").addEventListener("click", (event) => {{
  const freeze = event.target.textContent === "Freeze physics";
  networks.forEach((network) => {{
    network.setOptions({{ physics: {{ enabled: !freeze }} }});
  }});
  event.target.textContent = freeze ? "Unfreeze physics" : "Freeze physics";
}});

document.getElementById("color-by").addEventListener("change", (event) => {{
  const field = COLOR_FIELD[event.target.value];
  networks.forEach((network) => {{
    const updates = network.dataset.get().map((node) => ({{ id: node.id, color: node[field] }}));
    network.dataset.update(updates);
  }});
}});

document.querySelectorAll("#metrics th").forEach((header) => {{
  header.addEventListener("click", () => {{
    const table = document.getElementById("metrics");
    const body = table.tBodies[0];
    const index = header.cellIndex;
    const numeric = header.dataset.type === "num";
    const descending = header.dataset.dir !== "desc";
    const rows = Array.from(body.rows);
    rows.sort((left, right) => {{
      const a = left.cells[index].dataset.value;
      const b = right.cells[index].dataset.value;
      if (numeric) {{
        return descending ? Number(b) - Number(a) : Number(a) - Number(b);
      }}
      return descending ? b.localeCompare(a) : a.localeCompare(b);
    }});
    rows.forEach((row) => body.appendChild(row));
    table.querySelectorAll("th").forEach((other) => {{ other.dataset.dir = ""; }});
    header.dataset.dir = descending ? "desc" : "asc";
  }});
}});
</script>
</body>
</html>
"""


def _vis_script(vis_js: str) -> str:
    # The library is one script. The page script below it builds six networks.
    return vis_js


def network_options() -> dict:
    """Physics stays on, but stabilization stops early so six layouts can share a tab."""
    return {
        "physics": {
            "enabled": True,
            "barnesHut": {
                "gravitationalConstant": -4000,
                "centralGravity": 0.25,
                "springLength": 90,
                "springConstant": 0.04,
                "damping": 0.7,
                "avoidOverlap": 0.15,
            },
            "stabilization": {"enabled": True, "iterations": STABILIZATION_ITERATIONS, "fit": True},
        },
        "interaction": {"hover": True},
        # The improved layout pass walks every edge before physics starts. On these graphs it stalls the tab.
        "layout": {"improvedLayout": False},
        "edges": {"smooth": False, "color": "#d6d3d1", "width": 0.4},
        "nodes": {"font": {"size": 11, "color": "#44403c"}},
    }


def network_bundle(panels: list[dict], vis_js: str) -> tuple[str, str]:
    """Markup and scripts for the six graphs. The findings page embeds both."""
    cards = []
    data = []
    for index, panel in enumerate(panels):
        if index == 0:
            cards.append("<h2>Follow graphs, plus the SNAP friendship graph</h2>")
        if index == 3:
            cards.append("<h2>Co-participation</h2>")
        element = f"net-{index}"
        cards.append(
            "<section class=\"card\">"
            f"<p class=\"caption\">{panel['caption']}</p>"
            f"<div id=\"{element}\" class=\"network\"></div>"
            "</section>"
        )
        data.append({"element": element, "nodes": panel["nodes"], "edges": panel["edges"]})
    markup = (
        '<section id="networks" class="networks-wrap">'
        '<div class="toolbar">'
        '<input id="search" type="search" placeholder="Highlight a handle in all six graphs" autocomplete="off">'
        '<button id="physics-toggle" type="button">Freeze physics</button>'
        "<label>color by "
        '<select id="color-by">'
        '<option value="community">community</option>'
        '<option value="pagerank">pagerank</option>'
        '<option value="degree">degree</option>'
        "</select></label></div>"
        '<div class="networks">'
        + "".join(cards)
        + "</div></section>"
    )
    scripts = (
        "<script>\n"
        + _vis_script(vis_js)
        + "\n</script>\n<script>\n"
        + "const PANELS = "
        + _script_json(data)
        + ";\nconst OPTIONS = "
        + _script_json(network_options())
        + """;
const networks = [];
const COLOR_FIELD = { community: "colorCommunity", pagerank: "colorPagerank", degree: "colorDegree" };

function draw(panel) {
  const container = document.getElementById(panel.element);
  const data = {
    nodes: new vis.DataSet(panel.nodes),
    edges: new vis.DataSet(panel.edges),
  };
  const network = new vis.Network(container, data, OPTIONS);
  network.dataset = data.nodes;
  networks.push(network);
  // Stop the simulation once the short layout pass finishes, so opening the file does not pin the tab.
  network.once("stabilizationIterationsDone", () => {
    network.setOptions({ physics: false });
    const toggle = document.getElementById("physics-toggle");
    if (toggle) toggle.textContent = "Unfreeze physics";
  });
}

// Six Barnes-Hut starts at once stall the first paint. Stagger them by a frame.
PANELS.forEach((panel, index) => {
  setTimeout(() => draw(panel), index * 40);
});

function matchingIds(network, query) {
  const nodes = network.dataset.get();
  const folded = query.toLowerCase();
  const hits = nodes.filter((node) => {
    const label = String(node.label).toLowerCase();
    const id = String(node.id).toLowerCase();
    return label.includes(folded) || id.includes(folded);
  });
  // An exact handle wins over every node that merely contains those letters.
  const exact = hits.filter((node) => {
    const label = String(node.label).toLowerCase();
    const id = String(node.id).toLowerCase();
    return label === folded || id === folded || id.endsWith(":" + folded);
  });
  return (exact.length ? exact : hits).map((node) => node.id);
}

document.getElementById("search").addEventListener("input", (event) => {
  const query = event.target.value.trim();
  networks.forEach((network) => {
    if (!query) {
      network.unselectAll();
      return;
    }
    const ids = matchingIds(network, query);
    network.selectNodes(ids);
    if (ids.length) {
      network.fit({ nodes: ids, animation: false });
    }
  });
});

document.getElementById("physics-toggle").addEventListener("click", (event) => {
  const freeze = event.target.textContent === "Freeze physics";
  networks.forEach((network) => {
    network.setOptions({ physics: { enabled: !freeze } });
  });
  event.target.textContent = freeze ? "Unfreeze physics" : "Freeze physics";
});

document.getElementById("color-by").addEventListener("change", (event) => {
  const field = COLOR_FIELD[event.target.value];
  networks.forEach((network) => {
    const updates = network.dataset.get().map((node) => ({ id: node.id, color: node[field] }));
    network.dataset.update(updates);
  });
});
</script>
"""
    )
    return markup, scripts


def _modularity(graph: nx.Graph, communities: dict) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    try:
        # Same weight the analysis uses, so the caption matches the stored Louvain run.
        return float(_partition_record(graph, communities)["modularity"])
    except Exception:
        return float(_partition_record(graph, communities, weight=None)["modularity"])


def load_panels() -> list[dict]:
    """Read six runs. Metrics come from the stored health and Louvain rows."""
    loaded = []
    for spec in PANELS:
        meta = get_run(spec["run_id"])
        if meta is None:
            raise SystemExit(f"run {spec['run_id']} is not in the store")
        layer = (meta.get("config") or {}).get("layer")
        graph = load_graph(spec["run_id"], layer)
        if graph is None:
            raise SystemExit(f"run {spec['run_id']} has no graph")
        scores = align_scores(graph, load_metrics(spec["run_id"], "pagerank"))
        communities = align_communities(graph, load_communities(spec["run_id"], "louvain"))
        # Degree for the color scale is the full graph, not the capped picture.
        degrees = {node: int(degree) for node, degree in graph.degree()}
        health = load_result(spec["run_id"], "health") or {}
        network = health.get("network") or {}
        clustering = float(network.get("avg_clustering") or 0.0)
        assortativity = float(network.get("assortativity") or 0.0)
        modularity = _modularity(graph, communities)
        loaded.append(
            {
                "name": spec["name"],
                "family": spec["family"],
                "graph": graph,
                "scores": scores,
                "communities": communities,
                "degrees": degrees,
                "metrics": {
                    "clustering": clustering,
                    "assortativity": assortativity,
                    "modularity": modularity,
                    "halving": float(spec["halving"]),
                },
                "caption": caption_line(
                    spec["name"],
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                    modularity,
                    clustering,
                    float(spec["halving"]),
                ),
            }
        )
        print(spec["name"], graph.number_of_nodes(), graph.number_of_edges(), flush=True)
    return loaded


def view_panels(panels: list[dict], cap: int) -> list[dict]:
    """Replace each graph with its PageRank cap and the PyVis records for that cap."""
    views = []
    for panel in panels:
        view = cap_by_pagerank(panel["graph"], panel["scores"], cap)
        kept_scores = {node: panel["scores"][node] for node in view.nodes}
        kept_communities = {node: panel["communities"].get(node, 0) for node in view.nodes}
        kept_degrees = {node: panel["degrees"].get(node, 0) for node in view.nodes}
        nodes, edges = panel_records(view, kept_scores, kept_communities, kept_degrees)
        print(
            f"view {panel['name']}: {view.number_of_nodes()} nodes, {view.number_of_edges()} edges",
            flush=True,
        )
        views.append(
            {
                "name": panel["name"],
                "caption": panel["caption"],
                "metrics": panel["metrics"],
                "nodes": nodes,
                "edges": edges,
            }
        )
    return views


def build_html(panels: list[dict], cap: int, vis_js: str | None = None) -> str:
    source = _vis_source() if vis_js is None else vis_js
    return render_html(view_panels(panels, cap), source)


def write_page(path: Path = OUTPUT) -> Path:
    """Write the page. A file over 20 MB is rebuilt with 500 nodes per panel."""
    panels = load_panels()
    vis_js = _vis_source()
    html = render_html(view_panels(panels, NODE_CAP), vis_js)
    # The cap is the only size lever the page is allowed to pull.
    if len(html.encode("utf-8")) > MAX_BYTES:
        print(f"page is over {MAX_BYTES} bytes; redrawing at {NODE_CAP_FALLBACK}", flush=True)
        html = render_html(view_panels(panels, NODE_CAP_FALLBACK), vis_js)
    path.write_text(html, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return path


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()
