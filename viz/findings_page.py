"""One HTML page of the stored findings, in the same style as the platform comparison.

Run:
    python3 viz/findings_page.py

Numbers are the ones written in FINDINGS.md. The omega section is read from
results/reddit_halfyear_omega_sweep.json. Nothing is written back to the store.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "findings.html"
SWEEP_PATH = ROOT / "results" / "reddit_halfyear_omega_sweep.json"

# Louvain modularity of the stored partition. The rise is (2012 - 2008) / 2008.
MODULARITY = (
    {"run": "r2008-v2", "layer": "reddit_user", "nodes": 5110, "edges": 86268, "communities": 96, "modularity": 0.3246767820},
    {"run": "r2012-v2", "layer": "reddit_2012_user", "nodes": 54817, "edges": 556415, "communities": 1141, "modularity": 0.5582584329},
)
MODULARITY_RISE = (MODULARITY[1]["modularity"] - MODULARITY[0]["modularity"]) / MODULARITY[0]["modularity"]

# Communities that are not DISSOLVED. The other 78 have containment 0.
DISSOLUTION = (
    (0, 1210, 3, 0.095041, 108.44, 2, "SPLIT"),
    (1, 761, 4, 0.018397, 20.99, 4, "SPLIT"),
    (2, 665, 1, 0.024060, 27.45, 4, "SPLIT"),
    (3, 495, 4, 0.030303, 34.58, 4, "SPLIT"),
    (4, 318, 3, 0.047170, 53.82, 4, "SPLIT"),
    (5, 307, 4, 0.045603, 52.03, 3, "SPLIT"),
    (6, 302, 4, 0.029801, 34.00, 3, "SPLIT"),
    (7, 224, 4, 0.022321, 25.47, 4, "SPLIT"),
    (8, 146, 3, 0.041096, 46.89, 3, "SPLIT"),
    (9, 133, 3, 0.075188, 85.79, 2, "SPLIT"),
    (10, 106, 3, 0.056604, 64.58, 2, "SPLIT"),
    (11, 74, 1, 0.040541, 46.26, 2, "SPLIT"),
    (12, 70, 3, 0.042857, 48.90, 3, "SPLIT"),
    (13, 40, 0, 0.050000, 57.05, 2, "SPLIT"),
    (23, 3, 5, 0.333333, 380.33, 1, "MERGED"),
    (46, 2, 4, 0.500000, 570.50, 1, "MERGED"),
    (54, 2, 35, 0.500000, 570.50, 1, "STABLE"),
    (92, 2, 5, 0.500000, 570.50, 1, "MERGED"),
)
DISSOLUTION_COUNTS = (("DISSOLVED", 78), ("SPLIT", 14), ("MERGED", 3), ("STABLE", 1))

# Giant-component fraction after node removal.
REMOVAL_RATIOS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
ROBUSTNESS = (
    {"name": "2008 random", "color": "#78716c", "dash": "", "values": (0.948, 0.938, 0.907, 0.854, 0.751, 0.649)},
    {"name": "2008 degree", "color": "#b45309", "dash": "", "values": (0.942, 0.926, 0.879, 0.807, 0.632, 0.300)},
    {"name": "2008 betweenness", "color": "#1d4ed8", "dash": "", "values": (0.937, 0.917, 0.863, 0.768, 0.598, 0.280)},
    {"name": "2012 random", "color": "#a8a29e", "dash": "5 4", "values": (0.938, 0.928, 0.896, 0.843, 0.738, 0.633)},
    {"name": "2012 degree", "color": "#c2410c", "dash": "5 4", "values": (0.905, 0.877, 0.817, 0.703, 0.300, 0.022)},
    {"name": "2012 betweenness", "color": "#be123c", "dash": "5 4", "values": (0.900, 0.870, 0.791, 0.647, 0.132, 0.001)},
)

RICH_CLUB_K = (10, 20, 50, 100)
RICH_CLUB = (
    {"name": "2008", "color": "#1d4ed8", "values": (0.014804127170539385, 0.026138681710970866, 0.08923894763697328, 0.21477572559366753)},
    {"name": "2012", "color": "#be123c", "values": (0.0011088400418329738, 0.002226628821753049, 0.01125501295596993, 0.038446553565701834)},
)

CLOSENESS = (
    {"run": "r2008-v2", "n": 5110, "min": 0.0001957330, "median": 0.3361043755, "mean": 0.3222961113, "max": 0.4857467911, "who": "akdas"},
    {"run": "r2012-v2", "n": 54817, "min": 0.0000182428, "median": 0.2563272273, "mean": 0.2428832907, "max": 0.3966320935, "who": "incredible-ninja"},
)

FINGERPRINTS = (
    {"name": "antirez", "score": 0.9251, "klass": "UNIVERSAL", "github_rank": "5 / 1247", "reddit_rank": "377 / 5110"},
    {"name": "hadley", "score": 0.8497, "klass": "GITHUB-DOMINANT", "github_rank": "3 / 1247", "reddit_rank": "1637 / 5110"},
    {"name": "rtomayko", "score": 0.8109, "klass": "GITHUB-DOMINANT", "github_rank": "4 / 1247", "reddit_rank": "1615 / 5110"},
    {"name": "spez", "score": 0.7326, "klass": "GITHUB-DOMINANT", "github_rank": "7 / 1247", "reddit_rank": "516 / 5110"},
    {"name": "chromakode", "score": 0.6873, "klass": "GITHUB-DOMINANT", "github_rank": "2 / 1247", "reddit_rank": "3790 / 5110"},
)

PLATFORMS = (
    {"name": "GitHub follows", "nodes": 1247, "edges": 1542, "modularity": 0.6489554893, "assortativity": -0.8647454014896636, "clustering": 0.13118320762376973, "power": 0.7041682971997252, "halving": 0.01},
    {"name": "Bluesky follows", "nodes": 833, "edges": 915, "modularity": 0.6987064409, "assortativity": -0.95334050812306, "clustering": 0.02101115093398593, "power": 0.5409539362800933, "halving": 0.01},
    {"name": "SNAP Facebook", "nodes": 4039, "edges": 88234, "modularity": 0.8349209912, "assortativity": 0.06357722918564943, "clustering": 0.6055467186200862, "power": 0.8091782885710821, "halving": 0.30},
    {"name": "Reddit 2008", "nodes": 5110, "edges": 86268, "modularity": 0.3246767820, "assortativity": -0.016717553605958634, "clustering": 0.6401807425046512, "power": 0.863577291676034, "halving": 0.30},
    {"name": "GitHub co-contribution", "nodes": 733, "edges": 5400, "modularity": 0.7839194696, "assortativity": -0.10246224195867357, "clustering": 0.8794333679156484, "power": 0.592321643542107, "halving": 0.02},
    {"name": "Bluesky replies", "nodes": 1218, "edges": 12735, "modularity": 0.7379022935, "assortativity": -0.059616652774097334, "clustering": 0.9213323257263784, "power": 0.5263247371505854, "halving": 0.10},
)


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def _wrap_text(text: str, limit: int = 420) -> str:
    """Break prose so a generated line stays short enough for the preview."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) if not current else len(word) + 1
        if current and length + extra > limit:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def explain(means: str, matters: str) -> str:
    """The reading that sits under the stored numbers."""
    return (
        "<h3>What it means</h3>\n"
        f'<p class="note">{_wrap_text(esc(means))}</p>\n'
        "<h3>Why it matters</h3>\n"
        f'<p class="note">{_wrap_text(esc(matters))}</p>\n'
    )


def load_sweep(path: Path = SWEEP_PATH) -> list[dict]:
    """Persistence and modularity for each omega. Per-community rows are dropped."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    summary = []
    for row in rows:
        transitions = row["transitions"]
        summary.append(
            {
                "omega": float(row["omega"]),
                "modularity": float(row["modularity"]),
                "persistence": [float(item["mean_persistence"]) for item in transitions],
                "shared": [int(item["shared_members"]) for item in transitions],
                "labels": [f"{item['earlier']}→{item['later']}" for item in transitions],
                "communities": [int(item["communities"]) for item in row["snapshots"]],
                "nodes": [int(item["nodes"]) for item in row["snapshots"] if "nodes" in item],
                "months": list(row["months"]),
            }
        )
    return summary


def _chart_frame(width: int, height: int, y_max: float, y_ticks: int, x_labels: list[str], pad: tuple[int, int, int, int]) -> str:
    left, right, top, bottom = pad
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for step in range(y_ticks + 1):
        value = y_max * step / y_ticks
        y = top + plot_h * (1 - step / y_ticks)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{value:.2f}</text>')
    slot = plot_w / max(len(x_labels) - 1, 1)
    for index, label in enumerate(x_labels):
        x = left + index * slot
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="tick" text-anchor="middle">{esc(label)}</text>')
    return "\n".join(parts), left, top, plot_w, plot_h


def line_chart(series: tuple[dict, ...], x_labels: list[str], y_max: float, title: str) -> str:
    """A line chart. Dashed series are the later graph when both years are drawn."""
    width, height = 720, 280
    body, left, top, plot_w, plot_h = _chart_frame(width, height, y_max, 4, x_labels, (52, 16, 16, 32))
    slot = plot_w / max(len(x_labels) - 1, 1)
    lines = [body]
    for item in series:
        points = []
        for index, value in enumerate(item["values"]):
            x = left + index * slot
            y = top + plot_h * (1 - float(value) / y_max)
            points.append(f"{x:.1f},{y:.1f}")
        dash = f' stroke-dasharray="{item["dash"]}"' if item.get("dash") else ""
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{item["color"]}" stroke-width="2.4"{dash}/>'
        )
    lines.append("</svg>")
    legend = "\n".join(
        f'<span class="legend"><i style="background:{item["color"]}"></i>{esc(item["name"])}</span>'
        for item in series
    )
    return (
        f'<figure class="chart">\n<figcaption>{esc(title)}</figcaption>\n'
        + "\n".join(lines)
        + f'\n<div class="legend-row">{legend}</div>\n</figure>'
    )


def bar_chart(labels: list[str], values: list[float], captions: list[str], y_max: float, title: str, colors: list[str] | None = None) -> str:
    width, height = 720, 260
    left, right, top, bottom = 52, 16, 16, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    for step in range(5):
        value = y_max * step / 4
        y = top + plot_h * (1 - step / 4)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{value:.2f}</text>')
    slot = plot_w / max(len(labels), 1)
    bar_w = min(72, slot * 0.55)
    palette = colors or ["#44403c"] * len(labels)
    for index, (label, value, caption) in enumerate(zip(labels, values, captions)):
        x = left + index * slot + (slot - bar_w) / 2
        h = plot_h * (float(value) / y_max) if y_max else 0
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{palette[index % len(palette)]}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" class="tick" text-anchor="middle">{esc(caption)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 18}" class="tick" text-anchor="middle">{esc(label)}</text>')
    parts.append("</svg>")
    return (
        f'<figure class="chart">\n<figcaption>{esc(title)}</figcaption>\n'
        + "\n".join(parts)
        + "\n</figure>"
    )


def stacked_bar(pairs: tuple[tuple[str, int], ...], title: str) -> str:
    total = sum(count for _name, count in pairs)
    colors = {"DISSOLVED": "#a8a29e", "SPLIT": "#b45309", "MERGED": "#1d4ed8", "STABLE": "#15803d"}
    spans = []
    for name, count in pairs:
        width = 100 * count / total
        label = str(count) if width >= 8 else ""
        spans.append(
            f'<span style="width:{width:.4f}%;background:{colors[name]}" title="{esc(name)} {count}">{label}</span>'
        )
    legend = "\n".join(
        f'<span class="legend"><i style="background:{colors[name]}"></i>{esc(name)} {count}</span>'
        for name, count in pairs
    )
    return (
        f'<figure class="chart">\n<figcaption>{esc(title)}</figcaption>\n'
        f'<div class="stack" role="img">{"".join(spans)}</div>\n'
        f'<div class="legend-row">{legend}</div>\n</figure>'
    )


def table(headers: list[tuple[str, str]], rows: list[list[tuple[str, str]]]) -> str:
    """A sortable table. Each cell is (sort value, visible text)."""
    head = "".join(f'<th data-type="{kind}">{esc(label)}</th>' for label, kind in headers)
    body = []
    for row in rows:
        cells = "".join(f'<td data-value="{esc(value)}">{esc(text)}</td>' for value, text in row)
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap">\n<table class="sortable"><thead><tr>'
        + head
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body)
        + "\n</tbody></table>\n</div>"
    )


def section(anchor: str, number: str, title: str, claim: str, body: str) -> str:
    return (
        f'<section id="{anchor}">\n'
        f'<p class="eyebrow">{esc(number)}</p>\n'
        f"<h2>{esc(title)}</h2>\n"
        f'<p class="claim">{esc(claim)}</p>\n'
        f"{body}\n</section>"
    )


def _finding_modularity() -> str:
    earlier, later = MODULARITY
    chart = bar_chart(
        [earlier["run"], later["run"]],
        [earlier["modularity"], later["modularity"]],
        [f'{earlier["modularity"]:.3f}', f'{later["modularity"]:.3f}'],
        y_max=0.70,
        title="Louvain modularity",
        colors=["#1d4ed8", "#be123c"],
    )
    rows = [
        [
            (item["run"], item["run"]),
            (item["layer"], item["layer"]),
            (str(item["nodes"]), f'{item["nodes"]:,}'),
            (str(item["edges"]), f'{item["edges"]:,}'),
            (str(item["communities"]), f'{item["communities"]:,}'),
            (f'{item["modularity"]:.10f}', f'{item["modularity"]:.10f}'),
        ]
        for item in MODULARITY
    ]
    body = chart + table(
        [("run", "text"), ("layer", "text"), ("nodes", "num"), ("edges", "num"), ("communities", "num"), ("modularity", "num")],
        rows,
    )
    body += f'<p class="note">Rise = (0.5582584329 − 0.3246767820) / 0.3246767820 = {MODULARITY_RISE:.10f}.</p>'
    body += explain(
        "Modularity 0.325 in 2008 and 0.558 in 2012 means the later groups are more cleanly separated. "
        "In 2008 the co-participation graph is one overlapping mass. By 2012 people sit in tighter clusters.",
        "The network moved from a mixed crowd to distinct neighborhoods. "
        f"The rise is {MODULARITY_RISE:.3f}, about 72 percent.",
    )
    return section("finding-1", "Finding 1", "Modularity rose 72%", "Reddit co-participation in 2012 is more modular than the 2008 graph.", body)


def _finding_elites() -> str:
    body = (
        '<div class="stats">\n'
        '<p><strong>99</strong><span>of the 100 highest-PageRank accounts in 2012 are absent from 2008</span></p>\n'
        '<p><strong>1</strong><span>of the 2008 top 100 is still in the 2012 top 100: grauenwolf, rank 31 to rank 62</span></p>\n'
        '<p><strong>64</strong><span>of the 2008 top 100 are gone. 36 are still in the 2012 graph, and 1 is still in the top 100.</span></p>\n'
        "</div>\n"
        '<p class="note">compare.py --a r2008-v2 --b r2012-v2 --mode cohort --metric pagerank --top 100. '
        "grauenwolf PageRank share moved from 0.6067% to 0.1131%. "
        "Of the 36 who remain in the 2012 graph, 35 are no longer in the top 100.</p>\n"
    )
    body += explain(
        "The prominent accounts of 2008 did not merely slip a few ranks. "
        "64 of that top 100 are absent from the 2012 graph. 35 are still in the graph and out of the top 100. "
        "grauenwolf is the one account still in both top 100s, from rank 31 to rank 62.",
        "This is a 99 percent replacement of the 2012 hub list, and a 99 percent loss of the 2008 hub list. "
        "The accounts that defined the earlier graph are not the accounts that define the later one.",
    )
    return section("finding-2", "Finding 2", "Elite turnover 99%", "The 2012 hub list is almost a new set of accounts.", body)


def _finding_dissolution() -> str:
    chart = stacked_bar(DISSOLUTION_COUNTS, "96 communities from 2008, classified inside 2012")
    rows = [
        [
            (str(community), str(community)),
            (str(size), str(size)),
            (str(destination), str(destination)),
            (f"{containment:.6f}", f"{containment:.6f}"),
            (f"{lift:.2f}", f"{lift:.2f}"),
            (str(destinations), str(destinations)),
            (kind, kind),
        ]
        for community, size, destination, containment, lift, destinations, kind in DISSOLUTION
    ]
    body = chart + table(
        [
            ("2008 community", "num"),
            ("size", "num"),
            ("2012 community", "num"),
            ("containment", "num"),
            ("lift", "num"),
            ("destinations", "num"),
            ("class", "text"),
        ],
        rows,
    )
    body += (
        "<p class=\"note\">Null containment is 0.000876 (1/1141). "
        "The other 78 communities have containment 0, lift 0, and classification DISSOLVED. "
        "The one STABLE community is 54, a pair, with containment 0.500000 in 2012 community 35.</p>\n"
    )
    body += explain(
        "The 2008 communities did not grow into the 2012 ones. "
        "78 of 96 have containment 0. 14 split across more than one 2012 destination. "
        "3 small groups were absorbed. Community 54 is the stable case: size 2, destination 35, containment 0.5, lift 570.50.",
        "The group-level result matches the elite turnover. The 2008 partition does not survive inside 2012. "
        "Members left, or they no longer share a community.",
    )
    return section(
        "finding-3",
        "Finding 3",
        "Community dissolution",
        "78 of 96 communities from 2008 have no members left together in 2012.",
        body,
    )


def _finding_people() -> str:
    body = """
<div class="people">
<article>
<h3>akdas</h3>
<p class="tag">2008 only</p>
<ul>
<li>PageRank 0.004217, rank 1 / 5110</li>
<li>Degree 0.148561, rank 1</li>
<li>Betweenness 0.028257, rank 1</li>
<li>Closeness 0.485747, rank 1</li>
<li>Louvain 0, Leiden 1</li>
</ul>
</article>
<article>
<h3>grauenwolf</h3>
<p class="tag">both years</p>
<ul>
<li>PageRank 0.001665 rank 31 in 2008, 0.000370 rank 62 in 2012</li>
<li>Degree 0.063026 rank 33, then 0.008903 rank 54</li>
<li>Betweenness 0.003498 rank 106, then 0.003769 rank 81</li>
<li>Closeness 0.433624 rank 57, then 0.289038 rank 5443</li>
</ul>
</article>
<article>
<h3>incredible-ninja</h3>
<p class="tag">2012 only</p>
<ul>
<li>PageRank 0.001657, rank 1 / 54817</li>
<li>Degree 0.033913, rank 2</li>
<li>Betweenness 0.034073, rank 1</li>
<li>Closeness 0.396632, rank 1</li>
<li>Louvain 0, Leiden 0</li>
</ul>
</article>
</div>
"""
    body += explain(
        "akdas is rank 1 in 2008 on PageRank, degree, betweenness, and closeness, and is absent in 2012. "
        "grauenwolf stays in the top 100, rank 31 to rank 62, while closeness falls from rank 57 to rank 5443. "
        "incredible-ninja is absent in 2008 and, in 2012, rank 1 on PageRank, betweenness, and closeness, and rank 2 on degree.",
        "The three accounts are the turnover in miniature. "
        "The old center disappears, the one survivor keeps a high PageRank rank and loses reach, and the new center arrives already at the top.",
    )
    return section(
        "finding-4",
        "Finding 4",
        "Individual trajectories",
        "The 2008 top account is gone. The one elite who remains has fallen. The 2012 top account is new.",
        body,
    )


def _finding_robustness() -> str:
    labels = [f"{int(ratio * 100)}%" for ratio in REMOVAL_RATIOS]
    chart = line_chart(ROBUSTNESS, labels, 1.0, "Largest component, as a fraction of the intact graph")
    headers = [("ratio", "num")] + [(item["name"], "num") for item in ROBUSTNESS]
    rows = []
    for index, ratio in enumerate(REMOVAL_RATIOS):
        cells = [(f"{ratio:.2f}", labels[index])]
        for item in ROBUSTNESS:
            value = item["values"][index]
            cells.append((f"{value:.3f}", f"{value:.3f}"))
        rows.append(cells)
    note = (
        "<p class=\"note\">At 30% random removal the giant component is 0.649 in 2008 and 0.633 in 2012. "
        "At 30% degree removal it is 0.300 and 0.022. At 30% betweenness removal it is 0.280 and 0.001. "
        "That last cut leaves 15,989 components in 2012 and 1,029 in 2008.</p>\n"
    )
    note += explain(
        "Random deletion leaves both years about as connected: 0.649 of the 2008 giant component and 0.633 of the 2012 one after 30 percent. "
        "Deleting the highest-degree 30 percent leaves 0.300 in 2008 and 0.022 in 2012. "
        "Deleting the highest-betweenness 30 percent leaves 0.280 and 0.001.",
        "By 2012 the graph depends on a small set of hubs. "
        "Random loss is tolerated. Targeted removal of those hubs breaks the largest connected piece. "
        "That is the stored signature of a scale-free network, and it matches the thinner rich club and higher max degree in the next section.",
    )
    return section(
        "finding-5",
        "Finding 5",
        "Robustness to node removal",
        "2012 survives random removal about as well as 2008, and collapses when hubs are removed.",
        chart + table(headers, rows) + note,
    )


def _finding_structure() -> str:
    labels = [str(k) for k in RICH_CLUB_K]
    chart = line_chart(RICH_CLUB, labels, 0.25, "Rich-club coefficient")
    rows = [
        [("assortativity", "assortativity"), ("-0.0167175536", "-0.016718"), ("0.0006093685", "0.000609")],
        [("avg clustering", "avg clustering"), ("0.6401807425", "0.640181"), ("0.7401053405", "0.740105")],
        [("transitivity", "transitivity"), ("0.2000506008", "0.200051"), ("0.1692050724", "0.169205")],
        [("components", "components"), ("77", "77"), ("1082", "1,082")],
        [("avg degree", "avg degree"), ("33.7643835616", "33.764"), ("20.3008190890", "20.301")],
        [("max degree", "max degree"), ("759", "759"), ("1890", "1,890")],
        [("power-law alpha", "power-law alpha"), ("1.3470966438", "1.347"), ("1.8664661262", "1.866")],
        [("power-law R²", "power-law R²"), ("0.8635772917", "0.864"), ("0.8917107624", "0.892")],
    ]
    body = chart + table(
        [("metric", "text"), ("r2008-v2", "num"), ("r2012-v2", "num")],
        rows,
    )
    body += "<p class=\"note\">Both years fit a power law better than a lognormal or an exponential. The 2012 rich club is much thinner.</p>\n"
    body += explain(
        "Max degree rises from 759 to 1,890. Components rise from 77 to 1,082. "
        "The rich-club coefficient at 100 falls from 0.215 to 0.038. Power-law alpha rises from 1.347 to 1.866. "
        "The top accounts are larger, and they are less tied to each other.",
        "This is the structural change under the other Reddit results. "
        "The 2008 graph still has a connected high-degree core. The 2012 graph is a set of large broadcasters over a more fragmented graph.",
    )
    return section(
        "finding-6",
        "Finding 6",
        "Structural phase transition",
        "The 2012 graph has a higher max degree, more components, and a collapsed rich club.",
        body,
    )


def _finding_reach() -> str:
    labels = ["min", "median", "mean", "max"]
    # Two grouped readings drawn as two bar charts stacked in one figure via a line chart of the four summaries.
    series = tuple(
        {
            "name": item["run"],
            "color": "#1d4ed8" if item["run"] == "r2008-v2" else "#be123c",
            "dash": "" if item["run"] == "r2008-v2" else "5 4",
            "values": (item["min"], item["median"], item["mean"], item["max"]),
        }
        for item in CLOSENESS
    )
    chart = line_chart(series, labels, 0.50, "Closeness")
    rows = [
        [
            (item["run"], item["run"]),
            (str(item["n"]), f'{item["n"]:,}'),
            (f'{item["min"]:.10f}', f'{item["min"]:.6f}'),
            (f'{item["median"]:.10f}', f'{item["median"]:.6f}'),
            (f'{item["mean"]:.10f}', f'{item["mean"]:.6f}'),
            (f'{item["max"]:.10f}', f'{item["max"]:.6f}'),
            (item["who"], item["who"]),
        ]
        for item in CLOSENESS
    ]
    return section(
        "finding-7",
        "Finding 7",
        "Reach is flat",
        "Every node has a closeness score, and the 2012 maximum is lower than the 2008 maximum.",
        chart
        + table(
            [("run", "text"), ("n", "num"), ("min", "num"), ("median", "num"), ("mean", "num"), ("max", "num"), ("maximum account", "text")],
            rows,
        )
        + explain(
            "Closeness falls from 2008 to 2012. The maximum is 0.485747 for akdas and 0.396632 for incredible-ninja. "
            "The mean falls from 0.322296 to 0.242883. "
            "The graphs are disconnected, 77 components then 1,082, so this is reach inside that fragmented graph.",
            "The graph grew faster than its hubs' reach. "
            "A lower maximum closeness sits with the extra components: even the top account is less close to the rest of the graph.",
        ),
    )


def _finding_roles() -> str:
    ordered = sorted(FINGERPRINTS, key=lambda item: -item["score"])
    chart = bar_chart(
        [item["name"] for item in ordered],
        [item["score"] for item in ordered],
        [f'{item["score"]:.3f}' for item in ordered],
        y_max=1.0,
        title="Structural fingerprint",
        colors=["#15803d" if item["klass"] == "UNIVERSAL" else "#b45309" for item in ordered],
    )
    rows = [
        [
            (item["name"], item["name"]),
            (f'{item["score"]:.4f}', f'{item["score"]:.4f}'),
            (item["klass"], item["klass"]),
            (item["github_rank"].split()[0], item["github_rank"]),
            (item["reddit_rank"].split()[0], item["reddit_rank"]),
        ]
        for item in ordered
    ]
    note = (
        "<p class=\"note\">Class uses PageRank rank. Above the 90th percentile means rank/nodes ≤ 0.10. "
        "GitHub is github-multi-ego (1,247 nodes). Reddit is r2008-v2 (5,110 nodes). "
        "antirez is the only UNIVERSAL account. The other four are GITHUB-DOMINANT.</p>\n"
    )
    note += explain(
        "Fingerprint similarity is how close the two structural positions are. "
        "antirez scores 0.9251 and is high on both platforms. chromakode scores 0.6873: GitHub PageRank rank 2 of 1,247, Reddit PageRank rank 3,790 of 5,110. "
        "hadley is 3 and 1,637. rtomayko is 4 and 1,615. spez is 7 and 516.",
        "A GitHub rank does not predict the Reddit rank. "
        "The same person holds a different position once the edges mean something else.",
    )
    return section(
        "finding-8",
        "Finding 8",
        "Platform role divergence",
        "The same person can be central on GitHub and ordinary on Reddit.",
        chart
        + table(
            [("person", "text"), ("fingerprint", "num"), ("class", "text"), ("GitHub PageRank rank", "num"), ("Reddit PageRank rank", "num")],
            rows,
        )
        + note,
    )


def _finding_identity() -> str:
    body = """
<div class="people">
<article>
<h3>Within-graph z-score</h3>
<p>495 STRUCTURAL_ONLY rows, similarity 0.8014–0.988, covering all 100 Reddit hubs. 494 of them were the seven GitHub ego centers.</p>
</article>
<article>
<h3>Pooled z-score</h3>
<p>Still degenerate. Cosine among the seven GitHub centers was 0.966–1.000. No cross-platform pair cleared 0.8.</p>
</article>
<article>
<h3>Rank percentiles</h3>
<p>116,574 of 124,700 pairs (93.5%) scored above 0.8. Same-name ranks: antirez 266, hadley 898, chromakode 1115, rtomayko 1179, spez 1183. The best GitHub match for antirez on Reddit was marcel at 0.9839, not antirez at 0.9251.</p>
</article>
</div>
<p class="note">Seven features: degree, PageRank, betweenness, closeness, community size, mean neighbor degree, and neighbor-degree Gini. The structural cutoff stayed at 0.8. A shared name only adjusted confidence afterward.</p>
"""
    body += explain(
        "All three scalings fail as an identity method. "
        "Within-graph z-scores pile onto the same seven GitHub centers. "
        "Pooled z-scores make those centers look like each other and like nobody on Reddit. "
        "Rank percentiles put 93.5 percent of pairs above 0.8, and the best GitHub match for antirez on Reddit is marcel at 0.9839, not antirez at 0.9251.",
        "A structural fingerprint measures the difference in role. It does not recognize the same person across a follow graph and a co-participation graph. "
        "Cross-platform identity needs something besides these seven scores.",
    )
    return section(
        "finding-9",
        "Finding 9",
        "Fingerprinting fails across platforms",
        "Structure does not resolve identity when the edges mean different things.",
        body,
    )


def _finding_platforms() -> str:
    rows = [
        [
            (item["name"], item["name"]),
            (str(item["nodes"]), f'{item["nodes"]:,}'),
            (str(item["edges"]), f'{item["edges"]:,}'),
            (f'{item["modularity"]:.10f}', f'{item["modularity"]:.6f}'),
            (f'{item["assortativity"]:.10f}', f'{item["assortativity"]:.6f}'),
            (f'{item["clustering"]:.10f}', f'{item["clustering"]:.6f}'),
            (f'{item["power"]:.10f}', f'{item["power"]:.6f}'),
            (f'{item["halving"]:.2f}', f'{item["halving"]:.2f}'),
        ]
        for item in PLATFORMS
    ]
    short = ["GH follow", "Bsky follow", "SNAP", "Reddit 08", "GH co-contrib", "Bsky reply"]
    chart = bar_chart(
        short,
        [item["halving"] for item in PLATFORMS],
        [f'{item["halving"]:.2f}' for item in PLATFORMS],
        y_max=0.35,
        title="Degree-removal ratio where the largest component falls to half",
        colors=["#b45309", "#b45309", "#1d4ed8", "#1d4ed8", "#44403c", "#44403c"],
    )
    body = (
        chart
        + table(
            [
                ("platform", "text"),
                ("nodes", "num"),
                ("edges", "num"),
                ("modularity", "num"),
                ("assortativity", "num"),
                ("clustering", "num"),
                ("power-law R²", "num"),
                ("halving point", "num"),
            ],
            rows,
        )
        + '<p class="note">The drawings of these six graphs are in the <a href="#networks">Networks</a> section on this page. '
        "Follow graphs halve after removing 1% of high-degree nodes. Friendship and Reddit 2008 last until 30%.</p>\n"
        + explain(
            "Follow edges are interest in another account: GitHub clustering 0.131 and Bluesky clustering 0.021, both halving at 1 percent degree removal. "
            "Co-participation edges are shared rooms: Reddit 2008 clustering 0.640, Bluesky replies 0.921. "
            "SNAP friendship clusters at 0.606 and halves at 30 percent. "
            "GitHub changes shape with the edge rule: follows cluster at 0.131, co-contribution at 0.879.",
            "The shape follows the edge rule. The platform name does not. "
            "A follow graph is a set of broadcasters. A co-participation graph is a mesh. A friendship graph holds together under hub removal.",
        )
    )
    return section(
        "finding-10",
        "Finding 10",
        "Edge semantics determines network structure",
        "Follow graphs are fragile and disassortative. Co-participation graphs cluster. Friendship sits with Reddit on robustness.",
        body,
    )


def _finding_sweep(sweep: list[dict]) -> str:
    series = (
        {
            "name": "modularity",
            "color": "#1d4ed8",
            "dash": "",
            "values": tuple(item["modularity"] for item in sweep),
        },
        {
            "name": "mean persistence",
            "color": "#be123c",
            "dash": "",
            "values": tuple(sum(item["persistence"]) / len(item["persistence"]) for item in sweep),
        },
    )
    labels = [f'{item["omega"]:.1f}' for item in sweep]
    chart = line_chart(series, labels, 1.0, "Multislice modularity and mean persistence")
    headers = [("omega", "num"), ("modularity", "num")] + [(label, "num") for label in sweep[0]["labels"]]
    rows = []
    for item in sweep:
        cells = [(f'{item["omega"]:.1f}', f'{item["omega"]:.1f}'), (f'{item["modularity"]:.10f}', f'{item["modularity"]:.6f}')]
        for value in item["persistence"]:
            cells.append((f"{value:.6f}", f"{value:.0f}" if value in (0.0, 1.0) else f"{value:.6f}"))
        rows.append(cells)
    shared = sweep[0]["shared"]
    shared_text = ", ".join(f"{count:,}" for count in shared)
    nodes = sweep[0].get("nodes") or []
    overlap = ""
    if len(nodes) == len(shared) + 1 and all(count > 0 for count in nodes):
        rates = [shared[index] / nodes[index + 1] for index in range(len(shared))]
        overlap = (
            " Share of each later month already present in the earlier month: "
            + ", ".join(f"{rate:.1%}" for rate in rates)
            + "."
        )
    note = (
        f"<p class=\"note\">Shared people on the five pairs: {shared_text}. "
        "Omega 0.1, 0.5, 1.0, and 2.0 find the same communities. "
        "Community counts at those couplings are "
        + ", ".join(str(count) for count in sweep[1]["communities"])
        + " across "
        + ", ".join(sweep[0]["months"])
        + ". At omega 0 the counts are "
        + ", ".join(str(count) for count in sweep[0]["communities"])
        + "."
        + overlap
        + "</p>\n"
    )
    note += explain(
        "At omega 0, community ids are not aligned across months, so persistence is 0. "
        "From omega 0.1 through 2.0, every account present in both consecutive months keeps the same community id. "
        "Persistence 1.0 counts only those shared accounts."
        + overlap,
        "The people who appear in two consecutive months do not change community once the slices are coupled. "
        "The four-year change is the much larger set of accounts that are not in both graphs.",
    )
    return section(
        "multislice",
        "Half-year multislice",
        "Persistence is 0 at omega 0 and 1 from 0.1 up",
        "On six Reddit months, any coupling at or above 0.1 keeps every shared person in the same community.",
        chart + table(headers, rows) + note,
    )


def _steam() -> str:
    body = (
        "<p class=\"note\">steamcommunity.com/id/gabelogannewell reports “This profile is private.” "
        "The scraped layer was not usable, so it was removed.</p>\n"
        + explain(
            "SteamGPT returned 79 friends for Gabe Newell's account. Steam's own page for that account says the profile is private. "
            "The scrape is either cached from before the profile went private, or it is not the live profile. It is not a usable layer.",
            "The check against the live profile is why the other findings stay. "
            "They come from official APIs and public archives, and this one did not survive that check.",
        )
    )
    return section(
        "steam",
        "Verification",
        "Steam layer removed",
        "SteamGPT returned 79 friends for a profile Steam itself marks private.",
        body,
    )


def _story() -> str:
    return section(
        "story",
        "The whole story",
        "Edge meaning, not the platform",
        "Reddit from 2008 to 2012 went from a small connected graph to a large fragmented one.",
        explain(
            "The old elite left. The 2008 communities dissolved. "
            "Accounts present in consecutive half-year slices kept their community once coupling was on, and most of each later month was new. "
            "The 2012 graph has a higher max degree, more components, a thinner rich club, and it breaks when hubs are removed.",
            "Across the six graphs, the shape follows what an edge means. "
            "A follow is a broadcaster and an audience. A shared thread or a shared codebase is a mesh. A friendship holds under hub removal. "
            "The platform name does not decide which of those shapes you get.",
        ),
    )


def render_page(sweep: list[dict], networks_markup: str = "", network_scripts: str = "") -> str:
    """Full document. ``sweep`` is the omega summary from load_sweep.

    ``networks_markup`` and ``network_scripts`` are the six interactive graphs.
    An empty pair leaves the findings readable without them, which the unit test uses.
    """
    nav = []
    if networks_markup:
        nav.append(("networks", "Networks"))
    nav.extend((
        ("finding-1", "1 Modularity"),
        ("finding-2", "2 Elites"),
        ("finding-3", "3 Dissolution"),
        ("finding-4", "4 People"),
        ("finding-5", "5 Robustness"),
        ("finding-6", "6 Structure"),
        ("finding-7", "7 Reach"),
        ("finding-8", "8 Roles"),
        ("finding-9", "9 Identity"),
        ("finding-10", "10 Platforms"),
        ("multislice", "Multislice"),
        ("steam", "Steam"),
        ("story", "Story"),
    ))
    links = "".join(f'<a href="#{anchor}">{esc(label)}</a>' for anchor, label in nav)
    parts = [
        _finding_modularity(),
        _finding_elites(),
        _finding_dissolution(),
        _finding_people(),
        _finding_robustness(),
        _finding_structure(),
        _finding_reach(),
        _finding_roles(),
        _finding_identity(),
        _finding_platforms(),
        _finding_sweep(sweep),
        _steam(),
        _story(),
    ]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Findings</title>
<style>
body {{ margin: 0; font-family: "Iowan Old Style", Palatino, Georgia, serif; color: #1c1917; background: #fafaf9; }}
header, main, .networks-wrap {{ max-width: 1180px; margin: 0 auto; padding: 20px 24px 8px; }}
h1 {{ font-size: 32px; margin: 0 0 8px; }}
h2 {{ font-size: 24px; margin: 0 0 8px; }}
h3 {{ margin: 0 0 6px; font-size: 18px; }}
a {{ color: #1d4ed8; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px 14px; position: sticky; top: 0; background: #fafaf9; padding: 10px 0; z-index: 2; border-bottom: 1px solid #e7e5e4; }}
nav a {{ color: #44403c; text-decoration: none; font-size: 14px; }}
.lede, .claim, .note {{ line-height: 1.45; }}
.lede {{ margin-top: 0; }}
.eyebrow {{ margin: 28px 0 0; letter-spacing: 0.04em; text-transform: uppercase; font-size: 12px; color: #78716c; }}
.claim {{ margin-top: 0; }}
.note {{ color: #44403c; font-size: 15px; }}
table {{ border-collapse: collapse; background: #fff; width: 100%; }}
th, td {{ border: 1px solid #e7e5e4; padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
th {{ cursor: pointer; background: #f5f5f4; }}
th:first-child, td:first-child {{ text-align: left; }}
.table-wrap {{ overflow-x: auto; }}
.chart {{ margin: 12px 0 16px; }}
.chart figcaption {{ font-size: 14px; margin-bottom: 6px; }}
svg {{ width: 100%; height: auto; background: #fff; border: 1px solid #e7e5e4; }}
.grid {{ stroke: none; }}
.tick {{ font-size: 12px; fill: #44403c; font-family: "Iowan Old Style", Palatino, Georgia, serif; }}
line.grid {{ stroke: #e7e5e4; stroke-width: 1; }}
.legend-row {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 8px; font-size: 14px; }}
.legend {{ display: inline-flex; align-items: center; gap: 6px; }}
.legend i {{ width: 14px; height: 8px; display: inline-block; }}
.stack {{ display: flex; height: 36px; background: #fff; border: 1px solid #e7e5e4; color: #fff; font: 14px system-ui, sans-serif; }}
.stack span {{ display: flex; align-items: center; justify-content: center; overflow: hidden; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.stats p, .people article {{ background: #fff; border: 1px solid #e7e5e4; margin: 0; padding: 12px; }}
.stats strong {{ display: block; font-size: 36px; line-height: 1; }}
.people {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.people ul {{ margin: 0; padding-left: 18px; }}
.tag {{ margin: 0 0 8px; color: #78716c; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 12px 16px; align-items: center; padding: 4px 0 12px; }}
input, select, button {{ font: 16px system-ui, sans-serif; padding: 6px 8px; }}
.networks {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.networks h2 {{ grid-column: 1 / -1; font-size: 18px; margin: 8px 0; }}
.card {{ background: #fff; border: 1px solid #e7e5e4; min-width: 0; }}
.caption {{ margin: 0; padding: 8px 10px; font-size: 13px; line-height: 1.4; }}
.network {{ height: 420px; background: #fff; }}
.networks circle.dim {{ opacity: 0.15; }}
.networks circle.hit {{ stroke: #1c1917; stroke-width: 2px; }}
.networks text {{ font: 11px system-ui, sans-serif; fill: #44403c; pointer-events: none; }}
@media (max-width: 900px) {{
  .stats, .people, .networks {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header>
<h1>Findings</h1>
<p class="lede">Six graphs on four platforms: Reddit, GitHub, Bluesky, and Facebook. Dots are accounts and lines are relationships. The same measurements were run on each graph. Each section below is a stored result, then what that result means. The graphs are drawn on this page. Search highlights a handle in every one, and a column heading sorts a table.</p>
<nav>{links}</nav>
</header>
{networks_markup}
<main>
{"".join(parts)}
</main>
{network_scripts}
<script>
document.querySelectorAll("table.sortable th").forEach((header) => {{
  header.addEventListener("click", () => {{
    const table = header.closest("table");
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


# Drawings stay small enough for a browser and for Cursor's preview. The captions still
# report the full stored graph. vis.js is not embedded: that library is what made the
# downloaded file open as a broken preview.
DRAW_NODE_CAP = 100
DRAW_EDGE_CAP = 400


def _thin_edges(nodes: list[dict], edges: list[dict], limit: int) -> list[dict]:
    """Keep the edges that touch the highest-PageRank nodes. Size tracks PageRank."""
    if len(edges) <= limit:
        return edges
    rank = {node.get("id"): float(node.get("size") or 0.0) for node in nodes}

    def score(edge: dict) -> float:
        return rank.get(edge.get("from"), 0.0) + rank.get(edge.get("to"), 0.0)

    ranked = sorted(edges, key=score, reverse=True)
    return ranked[:limit]


def _layout(nodes: list[dict], edges: list[dict]) -> dict:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node.get("id") for node in nodes)
    for edge in edges:
        left, right = edge.get("from"), edge.get("to")
        if left in graph and right in graph:
            graph.add_edge(left, right)
    if graph.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(graph, seed=0, iterations=50)


def network_drawings(panels: list[dict]) -> tuple[str, str]:
    """Six SVG graphs plus a short search and color script. No bundled library."""
    cards = []
    for index, panel in enumerate(panels):
        if index == 0:
            cards.append("<h2>Follow graphs, plus the SNAP friendship graph</h2>")
        if index == 3:
            cards.append("<h2>Co-participation</h2>")
        nodes = list(panel["nodes"])
        edges = _thin_edges(nodes, list(panel["edges"]), DRAW_EDGE_CAP)
        pos = _layout(nodes, edges)
        xs = [point[0] for point in pos.values()] or [0.0]
        ys = [point[1] for point in pos.values()] or [0.0]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max_x - min_x or 1.0
        span_y = max_y - min_y or 1.0

        def place(node_id) -> tuple[float, float]:
            x, y = pos.get(node_id, (0.0, 0.0))
            return 36 + (x - min_x) / span_x * 928, 36 + (y - min_y) / span_y * 628

        parts = [
            f'<svg class="network" viewBox="0 0 1000 700" role="img" aria-label="{esc(panel["caption"])}">'
        ]
        for edge in edges:
            left, right = edge.get("from"), edge.get("to")
            if left not in pos or right not in pos:
                continue
            x1, y1 = place(left)
            x2, y2 = place(right)
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#d6d3d1" stroke-width="1"/>'
            )
        labeled = sorted(nodes, key=lambda node: float(node.get("size") or 0.0), reverse=True)[:8]
        labeled_ids = {node.get("id") for node in labeled}
        for node in nodes:
            node_id = node.get("id")
            if node_id not in pos:
                continue
            x, y = place(node_id)
            radius = 4.0 + 10.0 * (float(node.get("size") or 6.0) - 6.0) / 22.0
            radius = min(14.0, max(3.5, radius))
            label = str(node.get("label") or node_id)
            parts.append(
                "<circle "
                f'cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
                f'fill="{esc(node.get("color") or "#44403c")}" '
                f'data-label="{esc(label)}" '
                f'data-community="{esc(node.get("colorCommunity") or node.get("color") or "#44403c")}" '
                f'data-pagerank="{esc(node.get("colorPagerank") or "#44403c")}" '
                f'data-degree="{esc(node.get("colorDegree") or "#44403c")}">'
                f"<title>{esc(node.get('title') or label)}</title></circle>"
            )
            if node_id in labeled_ids:
                parts.append(f'<text x="{x + radius + 2:.1f}" y="{y + 3:.1f}">{esc(label)}</text>')
        parts.append("</svg>")
        cards.append(
            '<section class="card"><p class="caption">'
            + esc(panel["caption"])
            + "</p>\n"
            + "\n".join(parts)
            + "\n</section>"
        )
    markup = (
        '<section id="networks" class="networks-wrap">'
        '<div class="toolbar">\n'
        '<input id="search" type="search" placeholder="Highlight a handle in all six graphs" autocomplete="off">\n'
        "<label>color by "
        '<select id="color-by">\n'
        '<option value="community">community</option>\n'
        '<option value="pagerank">pagerank</option>\n'
        '<option value="degree">degree</option>\n'
        "</select></label></div>\n"
        '<p class="note">Each drawing is the 100 highest-PageRank people, with at most 400 edges. The caption above a drawing is the full stored graph.</p>\n'
        '<div class="networks">\n'
        + "\n".join(cards)
        + "\n</div></section>"
    )
    scripts = """<script>
document.getElementById("search").addEventListener("input", (event) => {
  const query = event.target.value.trim().toLowerCase();
  const circles = Array.from(document.querySelectorAll(".networks circle"));
  const exact = circles.filter((circle) => (circle.dataset.label || "").toLowerCase() === query);
  const hits = exact.length ? exact : circles.filter((circle) => (circle.dataset.label || "").toLowerCase().includes(query));
  const hitSet = new Set(hits);
  circles.forEach((circle) => {
    const on = !query || hitSet.has(circle);
    circle.classList.toggle("dim", Boolean(query) && !on);
    circle.classList.toggle("hit", Boolean(query) && on);
  });
});
document.getElementById("color-by").addEventListener("change", (event) => {
  document.querySelectorAll(".networks circle").forEach((circle) => {
    circle.setAttribute("fill", circle.dataset[event.target.value] || circle.getAttribute("fill"));
  });
});
</script>
"""
    return markup, scripts


def write_page(path: Path = OUTPUT, sweep_path: Path = SWEEP_PATH) -> Path:
    """Write the findings and the six graphs into one file."""
    from viz.compare_platforms import load_panels, view_panels

    sweep = load_sweep(sweep_path)
    views = view_panels(load_panels(), DRAW_NODE_CAP)
    markup, scripts = network_drawings(views)
    html_text = render_page(sweep, markup, scripts)
    path.write_text(html_text, encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    return path


def main() -> None:
    write_page()


if __name__ == "__main__":
    main()
