"""
src/dataset_statistics.py
=========================
Generates dataset inventory and comprehensive trip-level statistics across all available partitions.
Outputs:
- outputs/csv/dataset_inventory.csv
- outputs/csv/trip_statistics.csv
- outputs/csv/validation_report.csv
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.data_loader import (
    get_available_partitions,
    get_trip_ids,
    get_trip_file_paths,
    load_trip,
    load_track,
    load_nodes,
    load_arcs,
    load_route,
)
from src.data_validation import haversine_vectorized, validate_full_trip


def compute_dataset_inventory(output_dir: Path) -> pd.DataFrame:
    """
    Inventory every trip: file presence, file sizes, observation/element counts.
    Saves outputs/csv/dataset_inventory.csv
    """
    records = []
    partitions = get_available_partitions()

    for partition in partitions:
        trip_ids = get_trip_ids(partition)
        for trip_id in trip_ids:
            paths = get_trip_file_paths(trip_id, partition)
            record = {
                "partition": partition,
                "trip_id": trip_id,
                "track_exists": paths["track"].exists(),
                "nodes_exists": paths["nodes"].exists(),
                "arcs_exists": paths["arcs"].exists(),
                "route_exists": paths["route"].exists(),
                "track_size_bytes": paths["track"].stat().st_size if paths["track"].exists() else 0,
                "nodes_size_bytes": paths["nodes"].stat().st_size if paths["nodes"].exists() else 0,
                "arcs_size_bytes": paths["arcs"].stat().st_size if paths["arcs"].exists() else 0,
                "route_size_bytes": paths["route"].stat().st_size if paths["route"].exists() else 0,
            }
            record["total_size_bytes"] = (
                record["track_size_bytes"]
                + record["nodes_size_bytes"]
                + record["arcs_size_bytes"]
                + record["route_size_bytes"]
            )
            record["all_files_present"] = (
                record["track_exists"]
                and record["nodes_exists"]
                and record["arcs_exists"]
                and record["route_exists"]
            )

            # Load actual trip to count elements
            if record["all_files_present"]:
                trip = load_trip(trip_id, partition)
                record["n_gps_observations"] = len(trip["track"])
                record["n_nodes"] = len(trip["nodes"])
                record["n_arcs"] = len(trip["arcs"])
                record["n_route_entries"] = len(trip["route"])
            else:
                record["n_gps_observations"] = 0
                record["n_nodes"] = 0
                record["n_arcs"] = 0
                record["n_route_entries"] = 0

            records.append(record)

    df = pd.DataFrame(records)
    csv_path = output_dir / "dataset_inventory.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Inventory] Saved {len(df)} trip records to {csv_path}")
    return df


def compute_trip_statistics(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute detailed trip-wise statistics and validation report for all trips.
    Saves:
    - outputs/csv/trip_statistics.csv
    - outputs/csv/validation_report.csv
    """
    trip_stats = []
    val_records = []
    partitions = get_available_partitions()

    for partition in partitions:
        trip_ids = get_trip_ids(partition)
        for trip_id in trip_ids:
            trip = load_trip(trip_id, partition)

            # Validation
            val_res = validate_full_trip(trip)
            val_records.append(val_res)

            # Trajectory stats
            track = trip["track"]
            nodes = trip["nodes"]
            arcs = trip["arcs"]
            route = trip["route"]

            n_gps = len(track)
            timestamps = track["timestamp"].values
            duration_s = float(timestamps[-1] - timestamps[0]) if n_gps > 1 else 0.0

            dt = np.diff(timestamps) if n_gps > 1 else np.array([])
            pct_1s = float(np.mean(np.isclose(dt, 1.0, atol=1e-3)) * 100.0) if len(dt) > 0 else 100.0

            lons = track["longitude"].values
            lats = track["latitude"].values

            displacements = (
                haversine_vectorized(lons[:-1], lats[:-1], lons[1:], lats[1:])
                if n_gps > 1
                else np.array([])
            )
            total_dist_km = float(np.sum(displacements) / 1000.0) if len(displacements) > 0 else 0.0

            safe_dt = np.where(dt > 0, dt, 1e-6)
            speeds = (displacements / safe_dt) if len(displacements) > 0 else np.array([])
            mean_speed_kmh = float(np.mean(speeds) * 3.6) if len(speeds) > 0 else 0.0
            max_speed_kmh = float(np.max(speeds) * 3.6) if len(speeds) > 0 else 0.0

            # Route physical length (sum of arc lengths in route)
            route_length_km = 0.0
            if len(route) > 0:
                from_nodes = arcs["from_node_id"].values
                to_nodes = arcs["to_node_id"].values
                node_lons = nodes["longitude"].values
                node_lats = nodes["latitude"].values

                r_arcs = route.values
                valid_mask = (r_arcs >= 0) & (r_arcs < len(arcs))
                valid_r_arcs = r_arcs[valid_mask]

                u = from_nodes[valid_r_arcs]
                v = to_nodes[valid_r_arcs]

                arc_dists = haversine_vectorized(
                    node_lons[u], node_lats[u], node_lons[v], node_lats[v]
                )
                route_length_km = float(np.sum(arc_dists) / 1000.0)

            stats_record = {
                "partition": partition,
                "trip_id": trip_id,
                "n_gps_observations": n_gps,
                "duration_seconds": duration_s,
                "duration_minutes": round(duration_s / 60.0, 2),
                "pct_1s_intervals": round(pct_1s, 2),
                "total_gps_distance_km": round(total_dist_km, 3),
                "mean_speed_kmh": round(mean_speed_kmh, 2),
                "max_speed_kmh": round(max_speed_kmh, 2),
                "gps_min_lon": float(lons.min()),
                "gps_max_lon": float(lons.max()),
                "gps_min_lat": float(lats.min()),
                "gps_max_lat": float(lats.max()),
                "n_road_nodes": len(nodes),
                "n_road_arcs": len(arcs),
                "network_density": round(len(arcs) / max(1, len(nodes)), 3),
                "n_route_arcs": len(route),
                "route_length_km": round(route_length_km, 3),
                "route_continuity_pct": round(val_res["route_continuity_pct"], 2),
                "gps_to_route_ratio": round(n_gps / max(1, len(route)), 3),
                "issues_count": val_res["issues_count"],
            }
            trip_stats.append(stats_record)

    df_stats = pd.DataFrame(trip_stats)
    stats_csv = output_dir / "trip_statistics.csv"
    df_stats.to_csv(stats_csv, index=False)
    print(f"[Statistics] Saved {len(df_stats)} trip statistics to {stats_csv}")

    df_val = pd.DataFrame(val_records)
    val_csv = output_dir / "validation_report.csv"
    df_val.to_csv(val_csv, index=False)
    print(f"[Validation] Saved {len(df_val)} trip validation records to {val_csv}")

    return df_stats, df_val


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_dataset_inventory(out_dir)
    compute_trip_statistics(out_dir)
