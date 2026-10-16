"""Interactive HTML graph visualization with community-aware styling."""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx

try:
    from pyvis.network import Network
except ImportError:
    Network = None


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


NEGATIVE_LABELS = {"tense", "hostile", "condescending", "dismissive", "passive_aggressive"}
POSITIVE_LABELS = {"warm", "friendly"}

def _polarity_color(pos, neg, total, llm_polarity=None, llm_label=None):
    """Color by LLM label when present, fall back to lexicon polarity."""
    if llm_label:
        lbl = str(llm_label).strip().lower()
        if lbl in POSITIVE_LABELS:
            return "#3fa34d"
        if lbl in NEGATIVE_LABELS:
            return "#c43d3d"
        if lbl == "neutral":
            return "#888888"
    if llm_polarity is not None:
        p = float(llm_polarity)
    elif total == 0 or (pos + neg) == 0:
        return "#555555"
    else:
        p = (pos - neg) / max(pos + neg, 1)
    if p > 0.3:
        return "#3fa34d"
    if p < -0.3:
        return "#c43d3d"
    return "#888888"

def _is_llm_scored(data):
    return data.get("llm_polarity") is not None


def audit_communities(graph, min_size=3):
    communities = list(nx.community.louvain_communities(graph, seed=42))
    pr = nx.pagerank(graph, weight="weight")
    results = []
    for i, members in enumerate(communities):
        if len(members) < min_size:
            continue
        ms = set(members)
        n = len(members)
        max_internal = n * (n - 1) / 2
        internal = [(u, v) for u, v in graph.edges() if u in ms and v in ms]
        density = len(internal) / max_internal if max_internal else 0.0
        hub = max(members, key=lambda m: pr.get(m, 0))
        hub_edges = sum(1 for u, v in internal if hub in (u, v))
        hub_share = hub_edges / len(internal) if internal else 0.0
        others = set()
        for u, v in internal:
            if u != hub:
                others.add(u)
            if v != hub:
                others.add(v)
        if density > 0.3 and hub_share < 0.6:
            verdict = "real"
        elif hub_share > 0.75 and density < 0.15:
            verdict = "artifact"
        else:
            verdict = "mixed"
        results.append({"id": i, "members": members, "size": n, "hub": hub,
                        "density": density, "hub_share": hub_share,
                        "verdict": verdict})
    return results


def print_community_audit(graph):
    rows = audit_communities(graph)
    print(f"{'ID':<4}{'SIZE':<6}{'DENSITY':<10}{'HUB%':<8}{'VERDICT':<10}HUB")
    print("-" * 75)
    for r in rows:
        print(f"{r['id']:<4}{r['size']:<6}{r['density']:<10.3f}"
              f"{r['hub_share']:<8.2f}{r['verdict']:<10}{r['hub']}")


def _community_layout(graph, communities, node_comm, seed=42):
    comm_graph = nx.Graph()
    for i in range(len(communities)):
        comm_graph.add_node(i)
    for u, v in graph.edges():
        cu, cv = node_comm.get(u), node_comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        if comm_graph.has_edge(cu, cv):
            comm_graph[cu][cv]["weight"] += 1
        else:
            comm_graph.add_edge(cu, cv, weight=1)
    if len(comm_graph) > 1:
        comm_pos = nx.spring_layout(comm_graph, scale=4.0, seed=seed,
                                    weight="weight", k=3.0, iterations=200)
    else:
        comm_pos = {0: (0.0, 0.0)}
    pos = {}
    for comm_id, members in enumerate(communities):
        cx, cy = comm_pos.get(comm_id, (0.0, 0.0))
        n = len(members)
        if n == 1:
            pos[members[0]] = (cx, cy)
            continue
        radius = 0.6 + 0.08 * math.sqrt(n)
        for i, node in enumerate(members):
            angle = 2 * math.pi * i / n
            jitter_x = 0.05 * math.cos(i * 7.3)
            jitter_y = 0.05 * math.sin(i * 5.1)
            pos[node] = (cx + radius * math.cos(angle) + jitter_x,
                         cy + radius * math.sin(angle) + jitter_y)
    return pos


def _filter_edges(graph, top_n=120, min_weight=1.0):
    edges = [e for e in graph.edges(data=True) if e[2].get("weight", 1) >= min_weight]
    if top_n is not None and len(edges) > top_n:
        edges = sorted(edges, key=lambda e: -e[2].get("weight", 0))[:top_n]
    kept = nx.Graph()
    kept.add_nodes_from(graph.nodes(data=True))
    for u, v, d in edges:
        kept.add_edge(u, v, **d)
    return kept


def render_interactive_html(graph, output_path, height="900px",
                            top_edges=120, min_edge_weight=1.0,
                            top_labels=40):
    if Network is None:
        raise RuntimeError("pyvis is not installed. Run: pip3 install --user pyvis")

    full = graph
    vis = _filter_edges(graph, top_n=top_edges, min_weight=min_edge_weight)

    try:
        pr = nx.pagerank(full, weight="weight")
    except Exception:
        pr = {n: 1.0 for n in full.nodes()}

    values = list(pr.values()) or [1.0]
    lo, hi = min(values), max(values)
    spread = hi - lo if hi > lo else 1.0

    def size(n):
        norm = (pr.get(n, lo) - lo) / spread
        return int(18 + norm * 72)

    # Detect communities
    comm_list = []
    node_comm = {}
    try:
        comm_list = list(nx.community.louvain_communities(full, seed=42))
        for i, c in enumerate(comm_list):
            for node in c:
                node_comm[node] = i
    except Exception:
        pass

    # Audit each community for verdict
    audits = {}
    try:
        for a in audit_communities(full):
            audits[a["id"]] = a
    except Exception:
        pass

    positions = _community_layout(full, comm_list, node_comm) if comm_list else {}

    ranked = sorted(pr.items(), key=lambda kv: -kv[1])
    top_set = {n for n, _ in ranked[:top_labels]}

    net = Network(height=height, width="100%", bgcolor="#0f0f0f",
                  font_color="#f0f0f0", directed=False, cdn_resources="in_line")

    # Store per-node data for the click-to-isolate JS
    node_meta = {}

    for node in full.nodes():
        comm = node_comm.get(node)
        color = PALETTE[comm % len(PALETTE)] if comm is not None else "#888888"
        x, y = positions.get(node, (0.0, 0.0))
        attrs = full.nodes[node]
        msg_count = attrs.get("message_count", 0)
        verdict = audits.get(comm, {}).get("verdict", "?") if comm is not None else "?"
        tooltip = (f"<b>{node}</b><br>PageRank: {pr.get(node, 0):.4f}<br>"
                   f"Messages: {msg_count}<br>Community: {comm} ({verdict})")
        label = str(node) if node in top_set else ""
        net.add_node(node, label=label, title=tooltip,
                     x=float(x) * 400, y=float(y) * 400,
                     size=size(node), color=color, physics=False,
                     borderWidth=2 if verdict == "real" else (1 if verdict == "mixed" else 1),
                     shape="dot" if verdict == "real" else ("box" if verdict == "artifact" else "dot"),
                     font={"size": 18 if node in top_set else 12, "face": "arial",
                           "color": "#ffffff", "strokeWidth": 3,
                           "strokeColor": "#000000"},
                     group=comm)

    # Edge styling: bright if same community, dim if cross-community
    for u, v, data in vis.edges(data=True):
        weight = float(data.get("weight", 1.0))
        pos = int(data.get("positive_count", 0))
        neg = int(data.get("negative_count", 0))
        total = int(data.get("message_count", 0))

        cu = node_comm.get(u)
        cv = node_comm.get(v)
        same_comm = cu is not None and cu == cv

        lines = [f"<b>{u} \u2194 {v}</b>", f"Weight: {weight:.2f}"]
        if total:
            lines.append(f"Messages: {total}")
        if pos or neg:
            lines.append(f"Positive: {pos} &nbsp; Negative: {neg}")
            lines.append(f"Polarity: {(pos - neg) / max(pos + neg, 1):+.2f}")
        if not same_comm:
            lines.append("<i>cross-community edge</i>")
        if data.get("llm_label"):
            lines.append(f"LLM: <b>{data['llm_label']}</b>")

        edge_color = _polarity_color(pos, neg, total,
                                        llm_polarity=data.get("llm_polarity"),
                                        llm_label=data.get("llm_label"))
        is_llm = _is_llm_scored(data)
        # LLM-scored edges: solid, thick, real color
        if is_llm:
            edge_width = 3.0
            is_dashed = False
            lines.append("<i>LLM-scored edge</i>")
        else:
            # Unscored: show lexicon polarity, but dashed and thin
            edge_width = 1.0 if same_comm else 0.5
            is_dashed = True
            if not same_comm:
                edge_color = "#3a3a3a"
            lines.append("<i>lexicon-only (not yet LLM-scored)</i>")

        net.add_edge(u, v, title="<br>".join(lines), color=edge_color,
                     width=edge_width,
                     value=weight, physics=False,
                     dashes=is_dashed)

    net.set_options("""
    {
      "physics": {"enabled": false},
      "interaction": {"hover": true, "tooltipDelay": 50,
                      "navigationButtons": true, "dragNodes": true,
                      "dragView": true, "zoomView": true,
                      "multiselect": true, "selectConnectedEdges": true},
      "nodes": {"borderWidth": 2, "borderWidthSelected": 5},
      "edges": {"color": {"inherit": false, "highlight": "#ffffff"},
                "smooth": {"type": "continuous", "roundness": 0.3},
                "selectionWidth": 3}
    }
    """)

    out = Path(output_path)
    net.save_graph(str(out))
    _inject_extras(out, comm_list, pr, audits, vis, full, node_comm, top_set)
    return out


def _inject_extras(out, comm_list, pr, audits, vis, full, node_comm, top_set):
    html = out.read_text(encoding="utf-8")

    rows = []
    for i, c in enumerate(comm_list):
        color = PALETTE[i % len(PALETTE)]
        a = audits.get(i, {})
        best = a.get("hub", max(c, key=lambda n: pr.get(n, 0))) if c else "?"
        verdict = a.get("verdict", "?")
        icon = {"real": "&#9679;", "artifact": "&#9632;", "mixed": "&#9650;"}.get(verdict, "?")
        rows.append(
            f'<div class="osi-row"><span class="osi-dot" style="background:{color}">'
            f'</span>{len(c)} &middot; <b>{best}</b> <span style="color:#888">'
            f'[{verdict}]</span></div>'
        )

    legend = (
        '<div id="osi-legend" style="position:absolute;top:14px;left:14px;'
        'background:rgba(15,15,15,0.94);border:1px solid #333;border-radius:8px;'
        'padding:14px 16px;font-family:arial;font-size:13px;color:#eee;'
        'max-height:80vh;overflow:auto;z-index:9999">'
        '<div style="font-weight:bold;margin-bottom:8px;font-size:14px">Communities</div>'
        + "".join(rows) +
        '<div style="margin-top:14px;font-weight:bold;font-size:14px">Edge color</div>'
        '<div class="osi-row"><span class="osi-dot" style="background:#3fa34d">'
        '</span>positive (lexicon)</div>'
        '<div class="osi-row"><span class="osi-dot" style="background:#888888">'
        '</span>neutral (lexicon)</div>'
        '<div class="osi-row"><span class="osi-dot" style="background:#c43d3d">'
        '</span>negative</div>'
        '<div style="margin-top:10px;font-weight:bold;font-size:14px">Line style</div>'
        '<div class="osi-row"><span style="display:inline-block;width:22px;'
        'border-top:3px solid #888"></span>solid = LLM-scored</div>'
        '<div class="osi-row"><span style="display:inline-block;width:22px;'
        'border-top:2px dashed #555"></span>dashed = lexicon only</div>'
        '<div style="margin-top:14px;font-weight:bold;font-size:14px">Actions</div>'
        '<div class="osi-row" style="color:#9cf">Click a node to isolate its neighborhood</div>'
        '<div class="osi-row" style="color:#9cf">Double-click empty space to reset</div>'
        '<div style="margin-top:14px;font-weight:bold;font-size:14px">Shown</div>'
        f'<div class="osi-row">{vis.number_of_edges()} edges '
        f'(of {full.number_of_edges()})</div>'
        '</div>'
        '<style>'
        '.osi-row{display:flex;align-items:center;gap:8px;margin:3px 0}'
        '.osi-dot{width:14px;height:14px;border-radius:50%;display:inline-block;'
        'border:1px solid #555}'
        '</style>'
    )

    # JavaScript for click-to-isolate
    isolate_js = """
    <script>
    (function() {
      var allNodes = null;
      var allEdges = null;

      function waitForNetwork() {
        if (typeof network === 'undefined' || !network) {
          setTimeout(waitForNetwork, 100);
          return;
        }
        allNodes = network.body.data.nodes;
        allEdges = network.body.data.edges;
        var originalNodeColors = {};
        var originalEdgeColors = {};
        allNodes.forEach(function(n) { originalNodeColors[n.id] = n.color; });
        allEdges.forEach(function(e) { originalEdgeColors[e.id] = e.color; });

        network.on('click', function(params) {
          if (params.nodes.length === 0) return;
          var selected = params.nodes[0];
          var connected = new Set([selected]);
          var connectedEdges = new Set();

          allEdges.forEach(function(e) {
            if (e.from === selected || e.to === selected) {
              connected.add(e.from);
              connected.add(e.to);
              connectedEdges.add(e.id);
            }
          });

          var updates = [];
          allNodes.forEach(function(n) {
            var isConnected = connected.has(n.id);
            updates.push({
              id: n.id,
              color: isConnected ? originalNodeColors[n.id] : 'rgba(60,60,60,0.15)',
              font: isConnected
                ? {color: '#ffffff', strokeWidth: 3, strokeColor: '#000000'}
                : {color: 'rgba(200,200,200,0.2)', strokeWidth: 0}
            });
          });
          allEdges.forEach(function(e) {
            updates.push({
              id: e.id,
              color: connectedEdges.has(e.id) ? '#ffffff' : 'rgba(50,50,50,0.1)',
              width: connectedEdges.has(e.id) ? 3 : 0.5
            });
          });
          allNodes.update(updates);
        });

        network.on('doubleClick', function(params) {
          if (params.nodes.length > 0) return;
          var updates = [];
          allNodes.forEach(function(n) {
            updates.push({
              id: n.id,
              color: originalNodeColors[n.id],
              font: {color: '#ffffff', strokeWidth: 3, strokeColor: '#000000'}
            });
          });
          allEdges.forEach(function(e) {
            updates.push({id: e.id, color: originalEdgeColors[e.id]});
          });
          allNodes.update(updates);
          allEdges.update(updates);
        });
      }
      waitForNetwork();
    })();
    </script>
    """

    html = html.replace("<body>", "<body>" + legend + isolate_js)
    out.write_text(html, encoding="utf-8")
