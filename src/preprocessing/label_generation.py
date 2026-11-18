# src/preprocessing/label_generation.py
"""Label generation for PathSnap Phase 2.

This module aligns each GPS observation with the ground‑truth route arcs.
It produces a label (`candidate_arc_id`) if the candidate matches the true
arc for that observation, otherwise `-1`.  A confidence metric (distance to
true arc) is also returned.

The implementation does **not** modify any source files; it only reads the
pre‑processed candidate DataFrame and the route sequence.
"""

import pandas as pd
import numpy as np
from typing import Tuple

def _nearest_route_arc(observation: pd.Series, route_arc_ids: np.ndarray, arc_geometries: dict) -> Tuple[int, float]:
    """Find the nearest arc from the true route to a GPS observation.

    Parameters
    ----------
    observation: pd.Series
        Row containing `candidate_arc_id`, `gps_latitude`, `gps_longitude`, and
        `distance_to_candidate` (orthogonal distance).
    route_arc_ids: np.ndarray
        Ordered list of arc ids that constitute the ground‑truth route for the
        trip.
    arc_geometries: dict[int, Tuple[float, float, float, float]]
        Mapping from arc_id to its geometry as ``(lon1, lat1, lon2, lat2)``.

    Returns
    -------
    Tuple[int, float]
        The true arc id that is nearest to the observation and the orthogonal
        distance (meters) to that arc.
    """
    # Simple brute‑force search – route size is modest (few hundred arcs).
    # Compute orthogonal distance for each candidate arc geometry.
    obs_pt = (observation['gps_longitude'], observation['gps_latitude'])
    min_dist = np.inf
    nearest_arc = -1
    for arc_id in route_arc_ids:
        lon1, lat1, lon2, lat2 = arc_geometries[arc_id]
        # Compute perpendicular distance using vector projection.
        # Convert to meters using haversine approximations (fast).
        # Re‑use utility if available – fallback to simple Euclidean on lat/lon.
        # For speed we approximate 1 deg ≈ 111_320 m.
        # Vector from p1 to p2
        dx = lon2 - lon1
        dy = lat2 - lat1
        if dx == 0 and dy == 0:
            # degenerate segment – treat as point distance
            dist = np.sqrt((obs_pt[0] - lon1) ** 2 + (obs_pt[1] - lat1) ** 2) * 111_320
        else:
            # Parameter t of projection of point onto segment
            t = ((obs_pt[0] - lon1) * dx + (obs_pt[1] - lat1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            proj_lon = lon1 + t * dx
            proj_lat = lat1 + t * dy
            dist = np.sqrt((obs_pt[0] - proj_lon) ** 2 + (obs_pt[1] - proj_lat) ** 2) * 111_320
        if dist < min_dist:
            min_dist = dist
            nearest_arc = arc_id
    return nearest_arc, min_dist


def generate_labels(candidates_df: pd.DataFrame, route_df: pd.DataFrame, arc_geometries: dict) -> pd.DataFrame:
    """Assign labels to candidate rows.

    Parameters
    ----------
    candidates_df: pd.DataFrame
        DataFrame produced by ``candidate_generation``. Must contain columns:
        ``trip_id``, ``gps_index``, ``candidate_arc_id``, ``gps_longitude``,
        ``gps_latitude``.
    route_df: pd.DataFrame
        DataFrame with columns ``trip_id`` and ``arc_id`` (ordered by the
        route sequence).
    arc_geometries: dict[int, Tuple[float, float, float, float]]
        Geometry of each arc (lon1, lat1, lon2, lat2). Produced by
        ``road_preprocessing``.

    Returns
    -------
    pd.DataFrame
        Same as ``candidates_df`` with additional columns:
        * ``label`` – matching ``candidate_arc_id`` if it equals the nearest
          true arc, otherwise ``-1``.
        * ``true_arc_id`` – the nearest ground‑truth arc id.
        * ``distance_to_true_arc`` – orthogonal distance (m) to the true arc.
    """
    # Pre‑compute route sequences per trip for fast lookup
    route_by_trip = {}
    for trip_id, grp in route_df.groupby('trip_id'):
        route_by_trip[trip_id] = grp['arc_id'].values

    # Prepare output list
    records = []
    for _, row in candidates_df.iterrows():
        trip_id = row['trip_id']
        if trip_id not in route_by_trip:
            # No route info – mark as unlabeled
            true_arc = -1
            dist_true = np.nan
            label = -1
        else:
            true_arc, dist_true = _nearest_route_arc(row, route_by_trip[trip_id], arc_geometries)
            label = row['candidate_arc_id'] if row['candidate_arc_id'] == true_arc else -1
        out = row.to_dict()
        out.update({
            'label': int(label),
            'true_arc_id': int(true_arc),
            'distance_to_true_arc': float(dist_true) if not np.isnan(dist_true) else None,
        })
        records.append(out)
    return pd.DataFrame.from_records(records)
