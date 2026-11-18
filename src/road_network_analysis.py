"""
src/road_network_analysis.py
============================
Constructs road networks as NetworkX DiGraphs and analyzes topological properties:
- In-degree and Out-degree distributions
- Isolated nodes (zero total degree)
- Weakly connected components (WCC)
- Strongly connected components (SCC)
- Self-loops and multi-edges / duplicate arcs
- Road network density and reciprocity

Outputs:
- outputs/csv/road_network_statistics.csv
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import networkx as nx
import numpy as np
import pandas as pd

from src.data_loader import (
    get_available_partitions,
    get_trip_ids,
    load_trip,
    load_nodes,
    load_arcs,
)


def build_road_graph(nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> nx.DiGraph:
    """
    Construct a directed graph from nodes and arcs DataFrames.
    Preserves 0-based node_id as vertex identifier and stores longitude/latitude attributes.
    Preserves 0-based arc_id as edge attribute 'arc_id'.
    """
    G = nx.DiGraph()

    # Add nodes with geographical coordinates
    for node_id, row in nodes_df.iterrows():
        G.add_node(node_id, lon=row["longitude"], lat=row["latitude"])

    # Add directed edges with arc_id
    for arc_id, row in arcs_df.iterrows():
        u = int(row["from_node_id"])
        v = int(row["to_node_id"])
        # If edge already exists (duplicate directed edge), keep track
        G.add_edge(u, v, arc_id=arc_id)

    return G


def analyze_road_network_trip(trip_id: str, partition: str = "train", data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Analyze the road network of a single trip."""
    trip = load_trip(trip_id, partition, data_dir)
    nodes_df = trip["nodes"]
    arcs_df = trip["arcs"]

    n_nodes = len(nodes_df)
    n_arcs = len(arcs_df)

    from_nodes = arcs_df["from_node_id"].values
    to_nodes = arcs_df["to_node_id"].values

    # In-degree and out-degree fast array computation
    out_counts = np.bincount(from_nodes, minlength=n_nodes)
    in_counts = np.bincount(to_nodes, minlength=n_nodes)
    total_degrees = in_counts + out_counts

    isolated_nodes = int(np.sum(total_degrees == 0))
    source_nodes = int(np.sum((in_counts == 0) & (out_counts > 0)))
    sink_nodes = int(np.sum((in_counts > 0) & (out_counts == 0)))

    self_loops = int(np.sum(from_nodes == to_nodes))
    duplicate_arcs = int(arcs_df.duplicated(subset=["from_node_id", "to_node_id"]).sum())

    # Build NetworkX DiGraph for component analysis
    # For speed on large graphs, construct from edge list directly
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    edges = list(zip(from_nodes, to_nodes))
    G.add_edges_from(edges)

    n_wcc = nx.number_weakly_connected_components(G)
    wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
    largest_wcc_size = max(wcc_sizes) if wcc_sizes else 0
    wcc_coverage_pct = (largest_wcc_size / max(1, n_nodes)) * 100.0

    n_scc = nx.number_strongly_connected_components(G)
    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    largest_scc_size = max(scc_sizes) if scc_sizes else 0

    density = (n_arcs / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0

    return {
        "partition": partition,
        "trip_id": trip_id,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "in_degree_mean": round(float(np.mean(in_counts)), 3),
        "in_degree_max": int(np.max(in_counts)),
        "in_degree_median": int(np.median(in_counts)),
        "out_degree_mean": round(float(np.mean(out_counts)), 3),
        "out_degree_max": int(np.max(out_counts)),
        "out_degree_median": int(np.median(out_counts)),
        "isolated_nodes_count": isolated_nodes,
        "isolated_nodes_pct": round((isolated_nodes / max(1, n_nodes)) * 100.0, 3),
        "source_nodes_count": source_nodes,
        "sink_nodes_count": sink_nodes,
        "self_loops_count": self_loops,
        "duplicate_arcs_count": duplicate_arcs,
        "n_weakly_connected_components": n_wcc,
        "largest_wcc_size": largest_wcc_size,
        "largest_wcc_pct": round(wcc_coverage_pct, 2),
        "n_strongly_connected_components": n_scc,
        "largest_scc_size": largest_scc_size,
        "graph_density": density,
    }


def analyze_all_road_networks(output_dir: Path) -> pd.DataFrame:
    """Analyze road networks across all trips and save CSV."""
    records = []
    partitions = get_available_partitions()

    for partition in partitions:
        trip_ids = get_trip_ids(partition)
        print(f"[Road Network] Analyzing {len(trip_ids)} trips in partition '{partition}'...")
        for i, trip_id in enumerate(trip_ids):
            rec = analyze_road_network_trip(trip_id, partition)
            records.append(rec)
            if (i + 1) % 20 == 0 or (i + 1) == len(trip_ids):
                print(f"  Processed {i + 1}/{len(trip_ids)} trips in {partition}")

    df = pd.DataFrame(records)
    csv_path = output_dir / "road_network_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Road Network] Saved {len(df)} records to {csv_path}")
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_all_road_networks(out_dir)
