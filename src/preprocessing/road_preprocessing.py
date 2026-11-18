"""
src/preprocessing/road_preprocessing.py
=======================================
Road network preprocessing and MultiDiGraph representation:
- Preserves distinct arc IDs for duplicate / parallel directed edges
- Computes segment lengths, bearings, midpoints, and connectivity metrics
- Builds NetworkX MultiDiGraph (keyed by arc_id) to ensure no parallel edges are collapsed
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import networkx as nx

from src.utils.geo_utils import haversine_meters, forward_bearing_deg


def preprocess_road_network(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, nx.MultiDiGraph]:
    """
    Process road network nodes and arcs into an enriched arcs DataFrame and MultiDiGraph.
    Guarantees every original arc_id is preserved as the unique key.
    """
    arcs = arcs_df.copy()
    nodes = nodes_df.copy()

    from_nodes = arcs["from_node_id"].values
    to_nodes = arcs["to_node_id"].values

    # Endpoints coordinates
    from_lons = nodes.loc[from_nodes, "longitude"].values
    from_lats = nodes.loc[from_nodes, "latitude"].values
    to_lons = nodes.loc[to_nodes, "longitude"].values
    to_lats = nodes.loc[to_nodes, "latitude"].values

    arcs["from_lon"] = from_lons
    arcs["from_lat"] = from_lats
    arcs["to_lon"] = to_lons
    arcs["to_lat"] = to_lats

    # Midpoints
    arcs["mid_lon"] = 0.5 * (from_lons + to_lons)
    arcs["mid_lat"] = 0.5 * (from_lats + to_lats)

    # Segment length in meters and compass bearing [0, 360)
    arcs["segment_length"] = np.round(
        haversine_meters(from_lons, from_lats, to_lons, to_lats), 2
    )
    arcs["road_bearing"] = np.round(
        forward_bearing_deg(from_lons, from_lats, to_lons, to_lats), 2
    )

    # Topological connectivity
    n_nodes = len(nodes)
    out_degrees = np.bincount(from_nodes, minlength=n_nodes)
    in_degrees = np.bincount(to_nodes, minlength=n_nodes)

    arcs["from_out_degree"] = out_degrees[from_nodes]
    arcs["to_in_degree"] = in_degrees[to_nodes]

    # Detect reciprocity (presence of reverse edge (v, u))
    directed_pairs = set(zip(from_nodes, to_nodes))
    reverse_exists = np.array([
        (v, u) in directed_pairs for u, v in zip(from_nodes, to_nodes)
    ], dtype=bool)
    arcs["is_bidirectional"] = reverse_exists

    # Build NetworkX MultiDiGraph (each edge keyed by arc_id)
    G = nx.MultiDiGraph()
    for nid, row in nodes.iterrows():
        G.add_node(nid, lon=row["longitude"], lat=row["latitude"])

    for arc_id, row in arcs.iterrows():
        u = int(row["from_node_id"])
        v = int(row["to_node_id"])
        G.add_edge(
            u, v,
            key=arc_id,
            arc_id=arc_id,
            length=row["segment_length"],
            bearing=row["road_bearing"],
        )

    return arcs, G
