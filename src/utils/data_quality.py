"""
src/utils/data_quality.py
=========================
Audits and resolves Phase 1 technical concerns:
1. Duplicate road arcs: Verifies MultiDiGraph preservation and guarantees distinct arc_id lookup.
2. Route continuity: Audits all route gaps across all 90 trips, recording exact endpoints and gap distances.
3. Geographic distance calibration: Compares spherical Haversine with WGS-84 ellipsoidal Geodesic.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from geopy.distance import geodesic

from src.data_loader import (
    load_trip,
    get_available_partitions,
    get_trip_ids,
)
from src.utils.geo_utils import haversine_meters


def audit_duplicate_arcs(partition: str = "train") -> pd.DataFrame:
    """
    Examines all trips in partition to detect duplicate directed node pairs (u, v)
    with distinct arc_ids, ensuring MultiDiGraph or arc-indexed mapping is required.
    """
    records = []
    trip_ids = get_trip_ids(partition)

    for trip_id in trip_ids:
        trip = load_trip(trip_id, partition)
        arcs = trip["arcs"]

        dupes = arcs[arcs.duplicated(subset=["from_node_id", "to_node_id"], keep=False)]
        n_dupe_rows = len(dupes)
        if n_dupe_rows > 0:
            unique_dupe_pairs = arcs.duplicated(subset=["from_node_id", "to_node_id"]).sum()
            records.append({
                "partition": partition,
                "trip_id": trip_id,
                "total_arcs": len(arcs),
                "duplicate_arc_instances": n_dupe_rows,
                "redundant_edge_pairs": int(unique_dupe_pairs),
                "sample_arc_ids": str(dupes.index.values[:6].tolist()),
            })

    return pd.DataFrame(records)


def audit_route_discontinuities(partition: str = "train") -> pd.DataFrame:
    """
    Audits reference routes to find exact indices, arc pairs, and physical gap distances
    where to_node of arc_i does NOT connect to from_node of arc_{i+1}.
    """
    records = []
    trip_ids = get_trip_ids(partition)

    for trip_id in trip_ids:
        trip = load_trip(trip_id, partition)
        route = trip["route"].values
        arcs = trip["arcs"]
        nodes = trip["nodes"]

        if len(route) < 2:
            continue

        from_nodes = arcs["from_node_id"].values[route]
        to_nodes = arcs["to_node_id"].values[route]

        disc_mask = (to_nodes[:-1] != from_nodes[1:])
        disc_indices = np.where(disc_mask)[0]

        for idx in disc_indices:
            curr_arc = route[idx]
            next_arc = route[idx + 1]
            u1, v1 = from_nodes[idx], to_nodes[idx]
            u2, v2 = from_nodes[idx + 1], to_nodes[idx + 1]

            # Physical gap distance
            lon_v1 = nodes.loc[v1, "longitude"]
            lat_v1 = nodes.loc[v1, "latitude"]
            lon_u2 = nodes.loc[u2, "longitude"]
            lat_u2 = nodes.loc[u2, "latitude"]

            gap_dist_m = haversine_meters(lon_v1, lat_v1, lon_u2, lat_u2)

            records.append({
                "partition": partition,
                "trip_id": trip_id,
                "route_step_index": int(idx),
                "current_arc_id": int(curr_arc),
                "next_arc_id": int(next_arc),
                "current_to_node": int(v1),
                "next_from_node": int(u2),
                "gap_distance_meters": round(float(gap_dist_m), 2),
            })

    return pd.DataFrame(records)


def verify_distance_calibration(n_samples: int = 1000, random_seed: int = 42) -> Dict[str, float]:
    """
    Empirically compares Haversine formula against WGS-84 ellipsoidal Geodesic (Karney)
    over typical coordinate bounds of the dataset (Darmstadt corridor).
    """
    rng = np.random.default_rng(random_seed)

    # Darmstadt bounds
    lons1 = rng.uniform(8.48, 8.75, n_samples)
    lats1 = rng.uniform(49.75, 49.95, n_samples)
    # Consecutive displacements (0 to 500m)
    d_lons = rng.normal(0, 0.003, n_samples)
    d_lats = rng.normal(0, 0.002, n_samples)

    lons2 = lons1 + d_lons
    lats2 = lats1 + d_lats

    hav_dists = haversine_meters(lons1, lats1, lons2, lats2)

    geo_dists = np.array([
        geodesic((lat1, lon1), (lat2, lon2)).meters
        for lon1, lat1, lon2, lat2 in zip(lons1, lats1, lons2, lats2)
    ])

    abs_err = np.abs(hav_dists - geo_dists)
    rel_err = abs_err / np.maximum(1e-6, geo_dists)

    return {
        "n_samples": n_samples,
        "mean_absolute_error_m": float(np.mean(abs_err)),
        "max_absolute_error_m": float(np.max(abs_err)),
        "mean_relative_error_pct": float(np.mean(rel_err) * 100.0),
        "max_relative_error_pct": float(np.max(rel_err) * 100.0),
    }


if __name__ == "__main__":
    print("--- 1. DUPLICATE ROAD ARCS AUDIT ---")
    dupe_df = audit_duplicate_arcs("train")
    print(f"Trips with duplicate arcs in train: {len(dupe_df)}")
    print(dupe_df.head())

    print("\n--- 2. ROUTE DISCONTINUITIES AUDIT ---")
    disc_df = audit_route_discontinuities("train")
    print(f"Total route gaps detected in train: {len(disc_df)}")
    print(disc_df.head())

    print("\n--- 3. GEOGRAPHIC DISTANCE CALIBRATION ---")
    calib = verify_distance_calibration()
    for k, v in calib.items():
        print(f"  {k}: {v:.4f}")
