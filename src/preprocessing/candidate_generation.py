"""
src/preprocessing/candidate_generation.py
=========================================
Generates top-K candidate road segments for each GPS observation:
- Uses RoadNetworkSpatialIndex for O(log M) querying
- Calculates exact orthogonal perpendicular distance to candidate line segment
- Calculates angular difference between GPS heading and road segment bearing
- Preserves all original identifiers and ranks candidates 1 to K
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils.spatial_index import RoadNetworkSpatialIndex


def generate_trip_candidates(
    gps_df: pd.DataFrame,
    spatial_index: RoadNetworkSpatialIndex,
    trip_id: str,
    search_radius_m: float = 50.0,
    max_k: int = 10,
) -> pd.DataFrame:
    """
    Generate candidate road segments for all GPS observations of a trip.
    Returns:
        pd.DataFrame formatted strictly to the common Phase 3 feature schema.
    """
    records = []

    for _, row in gps_df.iterrows():
        g_idx = int(row["gps_index"])
        ts = float(row["timestamp"])
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        speed = float(row["gps_speed"])
        bearing = float(row["gps_bearing"])
        dt = float(row["time_difference"])
        is_outlier = bool(row["is_outlier"])
        is_stationary = bool(row["is_stationary"])
        dist_from_start = float(row["distance_from_start_km"])
        elapsed_ratio = float(row["elapsed_time_ratio"])

        # Query spatial index
        candidates = spatial_index.query_gps_point(
            gps_lon=lon,
            gps_lat=lat,
            gps_bearing=bearing,
            search_radius_m=search_radius_m,
            max_k=max_k,
        )

        for cand in candidates:
            records.append({
                "trip_id": trip_id,
                "gps_index": g_idx,
                "timestamp": ts,
                "gps_longitude": lon,
                "gps_latitude": lat,
                "candidate_arc_id": cand["candidate_arc_id"],
                "candidate_rank": cand["candidate_rank"],
                "distance_to_candidate": cand["distance_to_candidate"],
                "gps_speed": speed,
                "gps_bearing": bearing,
                "road_bearing": cand["road_bearing"],
                "bearing_difference": cand["bearing_difference"],
                "segment_length": cand["segment_length"],
                "time_difference": dt,
                "elapsed_time_ratio": elapsed_ratio,
                "distance_from_start_km": dist_from_start,
                "is_outlier": is_outlier,
                "is_stationary": is_stationary,
            })

    return pd.DataFrame(records)
