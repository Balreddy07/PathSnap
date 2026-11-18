"""
src/data_validation.py
======================
Comprehensive validation engine for PathSnap trips:
- GPS trajectory validation (.track)
- Road network node validation (.nodes)
- Road network arc validation (.arcs)
- Ground-truth route validation (.route)

Identifies and logs data quality issues, anomalies, discontinuities, and integrity violations
without mutating original records.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from geopy.distance import geodesic


def haversine_vectorized(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Compute geodesic distance in meters using the Haversine formula between coordinate arrays.
    Inputs are in decimal degrees.
    """
    R = 6371000.0  # Earth's radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return R * c


def validate_track(track_df: pd.DataFrame, trip_id: str = "") -> Dict[str, Any]:
    """Validate GPS observations (.track)."""
    n_points = len(track_df)
    issues = []

    # 1. Null / Non-numeric checks
    null_count = track_df.isnull().sum().to_dict()
    has_nulls = any(v > 0 for v in null_count.values())
    if has_nulls:
        issues.append(f"Contains null values: {null_count}")

    # 2. Coordinate range checks
    invalid_lon = ((track_df["longitude"] < -180.0) | (track_df["longitude"] > 180.0)).sum()
    invalid_lat = ((track_df["latitude"] < -90.0) | (track_df["latitude"] > 90.0)).sum()
    if invalid_lon > 0:
        issues.append(f"Invalid longitude count: {invalid_lon}")
    if invalid_lat > 0:
        issues.append(f"Invalid latitude count: {invalid_lat}")

    # 3. Duplicate checks
    exact_duplicates = track_df.duplicated().sum()
    coord_duplicates = track_df.duplicated(subset=["longitude", "latitude"]).sum()
    timestamp_duplicates = track_df.duplicated(subset=["timestamp"]).sum()
    if exact_duplicates > 0:
        issues.append(f"Exact duplicate rows: {exact_duplicates}")
    if timestamp_duplicates > 0:
        issues.append(f"Duplicate timestamps: {timestamp_duplicates}")

    # 4. Timestamp ordering and intervals
    timestamps = track_df["timestamp"].values
    dt = np.diff(timestamps) if n_points > 1 else np.array([])
    is_monotonic = bool(np.all(dt >= 0)) if len(dt) > 0 else True
    is_strictly_monotonic = bool(np.all(dt > 0)) if len(dt) > 0 else True
    negative_dt_count = int(np.sum(dt < 0)) if len(dt) > 0 else 0
    zero_dt_count = int(np.sum(dt == 0)) if len(dt) > 0 else 0
    large_dt_count = int(np.sum(dt > 5.0)) if len(dt) > 0 else 0  # gaps > 5s

    pct_exact_1s = float(np.mean(np.isclose(dt, 1.0, atol=1e-3)) * 100.0) if len(dt) > 0 else 100.0

    # 5. Consecutive point displacement & velocity
    displacements = np.array([])
    speeds = np.array([])
    large_jumps_count = 0  # velocity > 45 m/s (~162 km/h) or displacement > 200m in 1s

    if n_points > 1:
        lons = track_df["longitude"].values
        lats = track_df["latitude"].values
        displacements = haversine_vectorized(lons[:-1], lats[:-1], lons[1:], lats[1:])
        safe_dt = np.where(dt > 0, dt, 1e-6)
        speeds = displacements / safe_dt
        large_jumps_count = int(np.sum((speeds > 45.0) | (displacements > 250.0)))

    if large_jumps_count > 0:
        issues.append(f"Large GPS jumps / velocity anomalies count (>45 m/s or >250m): {large_jumps_count}")
    if not is_monotonic:
        issues.append(f"Non-monotonic timestamps count: {negative_dt_count}")

    return {
        "n_gps_points": n_points,
        "has_nulls": has_nulls,
        "invalid_coords_count": int(invalid_lon + invalid_lat),
        "exact_duplicates": int(exact_duplicates),
        "timestamp_duplicates": int(timestamp_duplicates),
        "coord_duplicates": int(coord_duplicates),
        "is_monotonic": is_monotonic,
        "is_strictly_monotonic": is_strictly_monotonic,
        "negative_dt_count": negative_dt_count,
        "zero_dt_count": zero_dt_count,
        "large_dt_count": large_dt_count,
        "pct_exact_1s": pct_exact_1s,
        "min_dt": float(np.min(dt)) if len(dt) > 0 else 0.0,
        "max_dt": float(np.max(dt)) if len(dt) > 0 else 0.0,
        "median_dt": float(np.median(dt)) if len(dt) > 0 else 0.0,
        "total_displacement_meters": float(np.sum(displacements)) if len(displacements) > 0 else 0.0,
        "max_speed_mps": float(np.max(speeds)) if len(speeds) > 0 else 0.0,
        "mean_speed_mps": float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
        "large_jumps_count": large_jumps_count,
        "issues": issues,
    }


def validate_nodes(nodes_df: pd.DataFrame, trip_id: str = "") -> Dict[str, Any]:
    """Validate road network nodes (.nodes)."""
    n_nodes = len(nodes_df)
    issues = []

    # 1. Null checks
    null_count = nodes_df.isnull().sum().to_dict()
    has_nulls = any(v > 0 for v in null_count.values())
    if has_nulls:
        issues.append(f"Nodes contains null values: {null_count}")

    # 2. Coordinates range
    invalid_lon = ((nodes_df["longitude"] < -180.0) | (nodes_df["longitude"] > 180.0)).sum()
    invalid_lat = ((nodes_df["latitude"] < -90.0) | (nodes_df["latitude"] > 90.0)).sum()
    if invalid_lon > 0 or invalid_lat > 0:
        issues.append(f"Invalid node coordinates count: lon={invalid_lon}, lat={invalid_lat}")

    # 3. Duplicate coordinates
    duplicate_nodes = int(nodes_df.duplicated().sum())

    return {
        "n_nodes": n_nodes,
        "has_nulls": has_nulls,
        "invalid_coords_count": int(invalid_lon + invalid_lat),
        "duplicate_node_coords": duplicate_nodes,
        "min_lon": float(nodes_df["longitude"].min()) if n_nodes > 0 else 0.0,
        "max_lon": float(nodes_df["longitude"].max()) if n_nodes > 0 else 0.0,
        "min_lat": float(nodes_df["latitude"].min()) if n_nodes > 0 else 0.0,
        "max_lat": float(nodes_df["latitude"].max()) if n_nodes > 0 else 0.0,
        "issues": issues,
    }


def validate_arcs(arcs_df: pd.DataFrame, n_nodes: int, trip_id: str = "") -> Dict[str, Any]:
    """Validate directed road network edges (.arcs)."""
    n_arcs = len(arcs_df)
    issues = []

    # 1. Null checks
    has_nulls = bool(arcs_df.isnull().any().any())

    # 2. Node references boundary checks
    out_of_bounds_from = ((arcs_df["from_node_id"] < 0) | (arcs_df["from_node_id"] >= n_nodes)).sum()
    out_of_bounds_to = ((arcs_df["to_node_id"] < 0) | (arcs_df["to_node_id"] >= n_nodes)).sum()
    invalid_node_refs = int(out_of_bounds_from + out_of_bounds_to)
    if invalid_node_refs > 0:
        issues.append(f"Invalid node references in arcs: {invalid_node_refs}")

    # 3. Self-loops (from_node == to_node)
    self_loops = int((arcs_df["from_node_id"] == arcs_df["to_node_id"]).sum())
    if self_loops > 0:
        issues.append(f"Self-loop arcs count: {self_loops}")

    # 4. Duplicate directed edges
    duplicate_arcs = int(arcs_df.duplicated(subset=["from_node_id", "to_node_id"]).sum())
    if duplicate_arcs > 0:
        issues.append(f"Duplicate directed arcs: {duplicate_arcs}")

    return {
        "n_arcs": n_arcs,
        "has_nulls": has_nulls,
        "invalid_node_refs": invalid_node_refs,
        "self_loops": self_loops,
        "duplicate_arcs": duplicate_arcs,
        "issues": issues,
    }


def validate_route(route_series: pd.Series, arcs_df: pd.DataFrame, trip_id: str = "") -> Dict[str, Any]:
    """Validate ground-truth reference route (.route)."""
    route_len = len(route_series)
    n_arcs = len(arcs_df)
    issues = []

    # 1. Null checks
    has_nulls = bool(route_series.isnull().any())
    if has_nulls:
        issues.append("Route contains null values")

    # 2. Arc ID range check
    route_arcs = route_series.values
    invalid_arc_ids = int(np.sum((route_arcs < 0) | (route_arcs >= n_arcs)))
    if invalid_arc_ids > 0:
        issues.append(f"Invalid arc IDs in route (out of bounds [0, {n_arcs-1}]): {invalid_arc_ids}")

    # 3. Consecutive arc connectivity
    discontinuous_transitions = 0
    valid_transitions = 0
    if route_len > 1 and invalid_arc_ids == 0:
        from_nodes = arcs_df["from_node_id"].values
        to_nodes = arcs_df["to_node_id"].values

        # For consecutive steps: arc_i and arc_{i+1}
        # A continuous transition occurs when to_node of arc_i == from_node of arc_{i+1}
        curr_to = to_nodes[route_arcs[:-1]]
        next_from = from_nodes[route_arcs[1:]]

        connected_mask = (curr_to == next_from)
        valid_transitions = int(np.sum(connected_mask))
        discontinuous_transitions = int(np.sum(~connected_mask))

        if discontinuous_transitions > 0:
            issues.append(f"Discontinuous route transitions: {discontinuous_transitions} of {route_len - 1}")

    continuity_pct = float((valid_transitions / max(1, route_len - 1)) * 100.0) if route_len > 1 else 100.0

    return {
        "route_length": route_len,
        "has_nulls": has_nulls,
        "invalid_arc_ids": invalid_arc_ids,
        "valid_transitions": valid_transitions,
        "discontinuous_transitions": discontinuous_transitions,
        "continuity_pct": continuity_pct,
        "unique_arcs_in_route": int(len(np.unique(route_arcs))) if route_len > 0 else 0,
        "issues": issues,
    }


def validate_full_trip(trip_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete validation across all 4 components of a single trip."""
    trip_id = trip_data["trip_id"]
    partition = trip_data.get("partition", "train")

    track_val = validate_track(trip_data["track"], trip_id)
    nodes_val = validate_nodes(trip_data["nodes"], trip_id)
    arcs_val = validate_arcs(trip_data["arcs"], nodes_val["n_nodes"], trip_id)
    route_val = validate_route(trip_data["route"], trip_data["arcs"], trip_id)

    all_issues = track_val["issues"] + nodes_val["issues"] + arcs_val["issues"] + route_val["issues"]

    return {
        "trip_id": trip_id,
        "partition": partition,
        "is_valid": len(all_issues) == 0,
        "issues_count": len(all_issues),
        "issues_summary": "; ".join(all_issues) if all_issues else "CLEAN",
        **{f"track_{k}": v for k, v in track_val.items() if k != "issues"},
        **{f"nodes_{k}": v for k, v in nodes_val.items() if k != "issues"},
        **{f"arcs_{k}": v for k, v in arcs_val.items() if k != "issues"},
        **{f"route_{k}": v for k, v in route_val.items() if k != "issues"},
    }
