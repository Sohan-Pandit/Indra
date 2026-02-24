"""
Stage 10 — Knowledge Graph + Visualization

Builds a directed NetworkX graph from extracted impact records and renders
it as an interactive HTML file using Pyvis.

Graph schema:
  Nodes: Hazard, Location, ImpactDomain, TimePeriod
  Edges: caused / contributed_to / associated_with / mitigated
         located_in / affected / co-occurred_with

Usage:
    python -m src.visualization.knowledge_graph \
        --records  data/weak_labels/labeled_abstracts.json \
        --output   outputs/impact_graph.html \
        --filter-hazard flood           # optional filter
        --filter-location "Bangladesh"  # optional filter
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Optional

import networkx as nx

try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except ImportError:
    _HAS_PYVIS = False


# ── Visual config ─────────────────────────────────────────────────────────────

NODE_COLORS: dict[str, str] = {
    "hazard":        "#e74c3c",   # red
    "location":      "#3498db",   # blue
    "impact_domain": "#2ecc71",   # green
    "time_period":   "#f39c12",   # orange
}

EDGE_COLORS: dict[str, str] = {
    "caused":           "#e74c3c",
    "contributed_to":   "#e67e22",
    "associated_with":  "#95a5a6",
    "mitigated":        "#2ecc71",
    "located_in":       "#3498db",
    "affected":         "#9b59b6",
    "co-occurred_with": "#1abc9c",
}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(records: list[dict]) -> nx.DiGraph:
    """
    Convert a list of ImpactRecord dicts into a directed NetworkX graph.

    Node weight = number of records mentioning that node.
    Edge weight = co-occurrence count.
    """
    G: nx.DiGraph = nx.DiGraph()

    def add_node(node_id: str, node_type: str) -> None:
        if node_id not in G:
            G.add_node(node_id, node_type=node_type, weight=0)
        G.nodes[node_id]["weight"] = G.nodes[node_id].get("weight", 0) + 1

    def add_edge(src: str, dst: str, relation: str) -> None:
        if G.has_edge(src, dst):
            G[src][dst]["weight"] = G[src][dst].get("weight", 0) + 1
        else:
            G.add_edge(src, dst, relation=relation, weight=1)

    for rec in records:
        hazard = rec.get("hazard_type")
        impact = rec.get("impact_domain")

        loc_dict = rec.get("location") or {}
        location = (loc_dict.get("normalized") or loc_dict.get("raw") or "").strip() or None

        time_dict = rec.get("time_period") or {}
        time_raw = time_dict.get("normalized") or time_dict.get("raw") or ""
        # Collapse to year for a cleaner graph
        time_node = time_raw[:4] if len(time_raw) >= 4 else time_raw or None

        causal = rec.get("causal_relation") or {}
        predicate = causal.get("predicate") or "associated_with"

        # Register nodes
        if hazard:
            add_node(hazard, "hazard")
        if impact:
            add_node(impact, "impact_domain")
        if location:
            add_node(location, "location")
        if time_node:
            add_node(time_node, "time_period")

        # Register edges
        if hazard and impact:
            add_edge(hazard, impact, predicate)
        if hazard and location:
            add_edge(hazard, location, "located_in")
        if impact and location:
            add_edge(impact, location, "affected")
        if hazard and time_node:
            add_edge(hazard, time_node, "co-occurred_with")

    return G


# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_graph(
    G: nx.DiGraph,
    hazard_type: Optional[str] = None,
    location: Optional[str] = None,
) -> nx.DiGraph:
    """
    Return a subgraph keeping only nodes reachable from the specified filter node.
    If no filter is supplied, the full graph is returned unchanged.
    """
    if not hazard_type and not location:
        return G

    seed_nodes: set[str] = set()

    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", "")
        if hazard_type and ntype == "hazard" and node_id == hazard_type:
            seed_nodes.add(node_id)
        if location and ntype == "location" and location.lower() in node_id.lower():
            seed_nodes.add(node_id)

    if not seed_nodes:
        return G

    reachable: set[str] = set(seed_nodes)
    for seed in seed_nodes:
        reachable.update(nx.descendants(G, seed))
        reachable.update(nx.ancestors(G, seed))

    return G.subgraph(reachable).copy()


# ── Pyvis rendering ───────────────────────────────────────────────────────────

_PHYSICS_OPTIONS = """
{
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "springLength": 160,
      "springConstant": 0.06,
      "damping": 0.4
    },
    "stabilization": {"iterations": 200}
  },
  "interaction": {
    "navigationButtons": true,
    "keyboard": true,
    "tooltipDelay": 100
  }
}
"""


def render_html(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Indra Knowledge Graph",
    bgcolor: str = "#1a1a2e",
    font_color: str = "white",
) -> str:
    """
    Render the NetworkX graph as an interactive Pyvis HTML file.

    Returns the absolute path to the written file.
    Raises ImportError if pyvis is not installed.
    """
    if not _HAS_PYVIS:
        raise ImportError(
            "pyvis is required for graph rendering. "
            "Install with: pip install pyvis"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor=bgcolor,
        font_color=font_color,
        heading="",  # title is shown by Streamlit, not duplicated inside the iframe
    )
    net.set_options(_PHYSICS_OPTIONS)

    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", "unknown")
        color = NODE_COLORS.get(ntype, "#cccccc")
        weight = attrs.get("weight", 1)
        size = 10 + min(weight * 4, 50)
        tooltip = f"<b>{node_id}</b><br>Type: {ntype}<br>Mentions: {weight}"
        net.add_node(
            str(node_id),
            label=str(node_id),
            color=color,
            size=size,
            title=tooltip,
            shape="dot",
        )

    for src, dst, attrs in G.edges(data=True):
        relation = attrs.get("relation", "associated_with")
        color = EDGE_COLORS.get(relation, "#95a5a6")
        width = min(1.0 + attrs.get("weight", 1) * 0.4, 8.0)
        net.add_edge(
            str(src),
            str(dst),
            title=relation,
            label=relation,
            color=color,
            width=width,
            arrows="to",
        )

    # Pyvis writes with system encoding on Windows; re-write as clean UTF-8
    net.write_html(str(output_path))
    raw = Path(output_path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    Path(output_path).write_text(text, encoding="utf-8")
    return str(output_path)


# ── Graph statistics ──────────────────────────────────────────────────────────

def graph_stats(G: nx.DiGraph) -> dict:
    """Return a summary statistics dict for the graph."""
    by_type: dict[str, list] = {}
    for node_id, attrs in G.nodes(data=True):
        ntype = attrs.get("node_type", "unknown")
        by_type.setdefault(ntype, []).append((node_id, attrs.get("weight", 0)))

    def top5(items: list) -> list[tuple]:
        return sorted(items, key=lambda x: -x[1])[:5]

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "nodes_by_type": {t: len(v) for t, v in by_type.items()},
        "top_hazards": top5(by_type.get("hazard", [])),
        "top_locations": top5(by_type.get("location", [])),
        "top_impacts": top5(by_type.get("impact_domain", [])),
        "density": round(nx.density(G), 4),
    }


# ── Convenience: build + render from file paths ───────────────────────────────

def build_and_render(
    records_path: Path,
    output_path: Path,
    hazard_filter: Optional[str] = None,
    location_filter: Optional[str] = None,
    title: str = "Indra Knowledge Graph",
) -> str:
    with open(records_path, encoding="utf-8") as f:
        records = json.load(f)

    G = build_graph(records)
    G = filter_graph(G, hazard_type=hazard_filter, location=location_filter)

    stats = graph_stats(G)
    print(f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    html_path = render_html(G, output_path, title=title)
    print(f"Rendered → {html_path}")
    return html_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and render Indra knowledge graph")
    parser.add_argument("--records", required=True, help="Labeled records JSON")
    parser.add_argument("--output", required=True, help="Output HTML path")
    parser.add_argument("--filter-hazard", default=None, dest="hazard")
    parser.add_argument("--filter-location", default=None, dest="location")
    args = parser.parse_args()

    build_and_render(
        records_path=Path(args.records),
        output_path=Path(args.output),
        hazard_filter=args.hazard,
        location_filter=args.location,
    )
