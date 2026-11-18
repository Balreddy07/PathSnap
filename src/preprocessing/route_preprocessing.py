"""
src/preprocessing/route_preprocessing.py
========================================
Reference route sequence validation and metadata extraction:
- Validates all arc IDs in .route against the road network
- Records continuity status and measures exact metric distances at discontinuities
- Preserves raw ground truth sequence without silent alteration
- Strict partition protection: Never fits learned transformations on test routes
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.utils.geo_utils import haversine_meters


def preprocess_route(
    route_s: pd.Series,
    arcs_processed_df: pd.DataFrame,
    trip_id: str,
) -> Dict[str, Any]:
    """
    Validate and extract topological summary of a ground-truth reference route.
    Does not modify or repair ground-truth records.
    """
    route_arcs = route_s.values
    route_len = len(route_arcs)
    n_arcs = len(arcs_processed_df)

    invalid_arcs = [int(a) for a in route_arcs if a < 0 or a >= n_arcs]
    if invalid_arcs:
        raise ValueError(f"Trip {trip_id} contains {len(invalid_arcs)} invalid arc IDs in route!")

    # Endpoints
    from_nodes = arcs_processed_df["from_node_id"].values[route_arcs]
    to_nodes = arcs_processed_df["to_node_id"].values[route_arcs]
    seg_lengths = arcs_processed_df["segment_length"].values[route_arcs]

    # Continuity
    discontinuities = []
    if route_len > 1:
        disc_mask = (to_nodes[:-1] != from_nodes[1:])
        disc_indices = np.where(disc_mask)[0]

        to_lons = arcs_processed_df["to_lon"].values[route_arcs]
        to_lats = arcs_processed_df["to_lat"].values[route_arcs]
        from_lons = arcs_processed_df["from_lon"].values[route_arcs]
        from_lats = arcs_processed_df["from_lat"].values[route_arcs]

        for idx in disc_indices:
            gap_m = haversine_meters(
                to_lons[idx], to_lats[idx],
                from_lons[idx + 1], from_lats[idx + 1]
            )
            discontinuities.append({
                "step_index": int(idx),
                "from_arc": int(route_arcs[idx]),
                "to_arc": int(route_arcs[idx + 1]),
                "gap_distance_m": round(float(gap_m), 2),
            })

    total_length_km = float(np.sum(seg_lengths) / 1000.0) if route_len > 0 else 0.0

    return {
        "trip_id": trip_id,
        "route_arc_count": route_len,
        "unique_arcs_count": int(len(np.unique(route_arcs))),
        "total_route_length_km": round(total_length_km, 3),
        "discontinuities_count": len(discontinuities),
        "discontinuities_list": discontinuities,
        "is_fully_continuous": len(discontinuities) == 0,
        "route_arcs_list": route_arcs.tolist(),
    }
