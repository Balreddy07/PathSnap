"""
src/utils/spatial_index.py
==========================
High-performance spatial index for road-network segments using scipy.spatial.cKDTree:
- Converts road arcs into metric 2D Cartesian plane (origin = trip centroid)
- Indexes segment midpoints in cKDTree for O(log M) spatial proximity querying
- Computes exact perpendicular point-to-line-segment distance (not midpoint approximation)
- Filters by search radius and returns top-K ranked road candidates per GPS observation
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.utils.geo_utils import (
    to_local_cartesian,
    point_to_segment_distance_and_proj,
    forward_bearing_deg,
    angular_difference_deg,
    haversine_meters,
)


class RoadNetworkSpatialIndex:
    """
    Spatial index for querying candidate road segments for GPS observations.
    Preserves all original arc IDs, including parallel / duplicate directed edges.
    """

    def __init__(self, arcs_df: pd.DataFrame, nodes_df: pd.DataFrame):
        self.n_arcs = len(arcs_df)
        self.arcs_df = arcs_df.copy()
        self.nodes_df = nodes_df.copy()

        # Extract coordinates
        from_nodes = arcs_df["from_node_id"].values
        to_nodes = arcs_df["to_node_id"].values

        self.from_lons = nodes_df.loc[from_nodes, "longitude"].values
        self.from_lats = nodes_df.loc[from_nodes, "latitude"].values
        self.to_lons = nodes_df.loc[to_nodes, "longitude"].values
        self.to_lats = nodes_df.loc[to_nodes, "latitude"].values

        # Arc midpoints
        self.mid_lons = 0.5 * (self.from_lons + self.to_lons)
        self.mid_lats = 0.5 * (self.from_lats + self.to_lats)

        # Arc physical lengths and bearings
        self.lengths_m = haversine_meters(
            self.from_lons, self.from_lats, self.to_lons, self.to_lats
        )
        self.bearings_deg = forward_bearing_deg(
            self.from_lons, self.from_lats, self.to_lons, self.to_lats
        )

        # Projection Origin (Centroid of nodes)
        self.origin_lon = float(nodes_df["longitude"].mean())
        self.origin_lat = float(nodes_df["latitude"].mean())

        # Project all endpoints and midpoints to local metric Cartesian coordinates
        self.ax, self.ay = to_local_cartesian(self.from_lons, self.from_lats, self.origin_lon, self.origin_lat)
        self.bx, self.by = to_local_cartesian(self.to_lons, self.to_lats, self.origin_lon, self.origin_lat)
        self.mid_x, self.mid_y = to_local_cartesian(self.mid_lons, self.mid_lats, self.origin_lon, self.origin_lat)

        # Midpoint KD-Tree
        midpoints_2d = np.column_stack((self.mid_x, self.mid_y))
        self.kdtree = cKDTree(midpoints_2d)

        # 95th percentile of half-lengths for safe search buffer
        self.max_half_length = float(np.percentile(self.lengths_m * 0.5, 99.0)) if self.n_arcs > 0 else 50.0

        # Store arc IDs as array for fast index retrieval
        self.arc_ids = arcs_df.index.values

    def query_gps_point(
        self,
        gps_lon: float,
        gps_lat: float,
        gps_bearing: Optional[float] = None,
        search_radius_m: float = 50.0,
        max_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query candidate road segments for a single GPS observation.
        Calculates exact perpendicular distance to each candidate segment.
        Returns sorted list of top-K candidate dictionaries.
        """
        px, py = to_local_cartesian(np.array([gps_lon]), np.array([gps_lat]), self.origin_lon, self.origin_lat)
        px_val, py_val = float(px[0]), float(py[0])

        # KDTree query radius: search_radius + max_half_length ensures no segment is missed
        effective_query_radius = search_radius_m + self.max_half_length
        candidate_indices = self.kdtree.query_ball_point([px_val, py_val], r=effective_query_radius)

        if not candidate_indices:
            # Fallback to nearest midpoint neighbor if ball query is empty
            _, nearest_idx = self.kdtree.query([px_val, py_val], k=1)
            candidate_indices = [int(nearest_idx)]

        # Calculate exact perpendicular point-to-segment distance for candidate pool
        evaluated_candidates = []
        for idx in candidate_indices:
            seg_dist, _, _ = point_to_segment_distance_and_proj(
                px_val, py_val,
                self.ax[idx], self.ay[idx],
                self.bx[idx], self.by[idx]
            )

            road_brg = float(self.bearings_deg[idx])
            bearing_diff = (
                float(angular_difference_deg(gps_bearing, road_brg))
                if gps_bearing is not None and not np.isnan(gps_bearing)
                else 0.0
            )

            evaluated_candidates.append({
                "candidate_arc_id": int(self.arc_ids[idx]),
                "distance_to_candidate": round(float(seg_dist), 2),
                "road_bearing": round(road_brg, 2),
                "bearing_difference": round(bearing_diff, 2),
                "segment_length": round(float(self.lengths_m[idx]), 2),
                "from_node_id": int(self.arcs_df.iloc[idx]["from_node_id"]),
                "to_node_id": int(self.arcs_df.iloc[idx]["to_node_id"]),
            })

        # Sort strictly by perpendicular segment distance
        evaluated_candidates.sort(key=lambda c: c["distance_to_candidate"])

        # Filter candidates exceeding search_radius unless all candidates exceed it
        filtered = [c for c in evaluated_candidates if c["distance_to_candidate"] <= search_radius_m]
        if not filtered:
            filtered = evaluated_candidates[:1]  # Keep closest to avoid empty candidate set

        # Truncate to max_k and assign 1-based ranks
        top_k = filtered[:max_k]
        for rank, cand in enumerate(top_k, start=1):
            cand["candidate_rank"] = rank

        return top_k
