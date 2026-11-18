"""
src/route_analysis.py
=====================
Detailed analysis of ground-truth reference routes (.route):
- Recovers edge endpoints from .arcs and coordinates from .nodes
- Validates arc IDs and connectivity between consecutive arcs (to_node of arc_i == from_node of arc_{i+1})
- Measures transition gaps (geodesic distance in meters when discontinuous)
- Compares route arc count with GPS observation count
- Documents route-data limitations and physical plausibility

Outputs:
- outputs/csv/route_statistics.csv
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.data_loader import (
    get_available_partitions,
    get_trip_ids,
    load_trip,
)
from src.data_validation import haversine_vectorized


def analyze_trip_route(trip_id: str, partition: str = "train", data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Perform in-depth analysis of a single trip's reference route."""
    trip = load_trip(trip_id, partition, data_dir)
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    route_len = len(route)
    n_gps = len(track)
    n_arcs = len(arcs)
    n_nodes = len(nodes)

    r_arcs = route.values
    invalid_arc_count = int(np.sum((r_arcs < 0) | (r_arcs >= n_arcs)))

    from_nodes = arcs["from_node_id"].values
    to_nodes = arcs["to_node_id"].values
    node_lons = nodes["longitude"].values
    node_lats = nodes["latitude"].values

    valid_mask = (r_arcs >= 0) & (r_arcs < n_arcs)
    valid_r_arcs = r_arcs[valid_mask]

    # Endpoints
    route_from = from_nodes[valid_r_arcs]
    route_to = to_nodes[valid_r_arcs]

    # Arc physical lengths
    arc_lengths = haversine_vectorized(
        node_lons[route_from], node_lats[route_from],
        node_lons[route_to], node_lats[route_to]
    )
    total_route_length_m = float(np.sum(arc_lengths)) if len(arc_lengths) > 0 else 0.0

    # Consecutive connectivity checks
    valid_transitions = 0
    discontinuous_transitions = 0
    max_gap_distance_m = 0.0
    mean_gap_distance_m = 0.0
    gap_distances = []

    if len(valid_r_arcs) > 1:
        curr_to = route_to[:-1]
        next_from = route_from[1:]
        is_connected = (curr_to == next_from)
        valid_transitions = int(np.sum(is_connected))
        discontinuous_transitions = int(np.sum(~is_connected))

        if discontinuous_transitions > 0:
            disc_indices = np.where(~is_connected)[0]
            # Calculate gap distance between disconnected endpoints
            gaps = haversine_vectorized(
                node_lons[curr_to[disc_indices]], node_lats[curr_to[disc_indices]],
                node_lons[next_from[disc_indices]], node_lats[next_from[disc_indices]]
            )
            gap_distances = gaps.tolist()
            max_gap_distance_m = float(np.max(gaps))
            mean_gap_distance_m = float(np.mean(gaps))

    continuity_pct = float((valid_transitions / max(1, len(valid_r_arcs) - 1)) * 100.0) if len(valid_r_arcs) > 1 else 100.0
    unique_arcs = int(len(np.unique(valid_r_arcs)))
    duplicate_arcs_in_route = int(route_len - unique_arcs)

    # GPS Trajectory total length for comparison
    gps_lons = track["longitude"].values
    gps_lats = track["latitude"].values
    gps_displacements = (
        haversine_vectorized(gps_lons[:-1], gps_lats[:-1], gps_lons[1:], gps_lats[1:])
        if n_gps > 1
        else np.array([])
    )
    total_gps_distance_m = float(np.sum(gps_displacements)) if len(gps_displacements) > 0 else 0.0

    length_diff_m = total_route_length_m - total_gps_distance_m
    length_ratio = (total_route_length_m / max(1.0, total_gps_distance_m))

    return {
        "partition": partition,
        "trip_id": trip_id,
        "route_arc_count": route_len,
        "unique_arcs_count": unique_arcs,
        "duplicate_arcs_in_route": duplicate_arcs_in_route,
        "invalid_arc_count": invalid_arc_count,
        "total_route_length_meters": round(total_route_length_m, 2),
        "total_route_length_km": round(total_route_length_m / 1000.0, 3),
        "mean_arc_length_meters": round(float(np.mean(arc_lengths)), 2) if len(arc_lengths) > 0 else 0.0,
        "max_arc_length_meters": round(float(np.max(arc_lengths)), 2) if len(arc_lengths) > 0 else 0.0,
        "valid_transitions_count": valid_transitions,
        "discontinuous_transitions_count": discontinuous_transitions,
        "continuity_pct": round(continuity_pct, 2),
        "max_gap_distance_meters": round(max_gap_distance_m, 2),
        "mean_gap_distance_meters": round(mean_gap_distance_m, 2),
        "n_gps_observations": n_gps,
        "total_gps_distance_meters": round(total_gps_distance_m, 2),
        "route_to_gps_distance_ratio": round(length_ratio, 3),
        "route_to_gps_count_ratio": round(route_len / max(1, n_gps), 3),
    }


def analyze_all_routes(output_dir: Path) -> pd.DataFrame:
    """Analyze reference routes across all trips and save CSV."""
    records = []
    partitions = get_available_partitions()

    for partition in partitions:
        trip_ids = get_trip_ids(partition)
        for trip_id in trip_ids:
            rec = analyze_trip_route(trip_id, partition)
            records.append(rec)

    df = pd.DataFrame(records)
    csv_path = output_dir / "route_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Route Analysis] Saved {len(df)} records to {csv_path}")
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    analyze_all_routes(out_dir)
