"""
src/utils/geo_utils.py
======================
High-performance vectorized geospatial calculations:
- Haversine great-circle distance (meters)
- Forward compass bearing (degrees in [0, 360))
- Minimal angular discrepancy between bearings (degrees in [0, 180])
- Metric local equirectangular coordinate projection centered at trip centroid
- Exact orthogonal perpendicular distance from a 2D point to a line segment (meters)
"""

from typing import Tuple, Union
import numpy as np


EARTH_RADIUS_METERS = 6371000.0


def haversine_meters(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute great-circle distance in meters between WGS-84 coordinate points.
    Supports scalars or NumPy arrays. Coordinates in decimal degrees.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return EARTH_RADIUS_METERS * c


def forward_bearing_deg(
    lon1: Union[float, np.ndarray],
    lat1: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute initial compass bearing (heading) from point 1 to point 2 in degrees [0, 360).
    0 = North, 90 = East, 180 = South, 270 = West.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


def angular_difference_deg(
    bearing1: Union[float, np.ndarray],
    bearing2: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Compute the minimal angular difference between two compass bearings in degrees [0, 180].
    Correctly accounts for circular wrap-around at 0/360 degrees.
    """
    diff = np.abs(bearing1 - bearing2) % 360.0
    return np.where(diff > 180.0, 360.0 - diff, diff)


def to_local_cartesian(
    lons: np.ndarray,
    lats: np.ndarray,
    origin_lon: float,
    origin_lat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project WGS-84 coordinates to local 2D metric Cartesian plane (x=East, y=North in meters)
    centered at (origin_lon, origin_lat) using conformal Equirectangular projection.
    Highly accurate (<0.05% error) within regional bounding boxes (<100 km).
    """
    phi0 = np.radians(origin_lat)
    x = np.radians(lons - origin_lon) * EARTH_RADIUS_METERS * np.cos(phi0)
    y = np.radians(lats - origin_lat) * EARTH_RADIUS_METERS
    return x, y


def point_to_segment_distance_and_proj(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> Tuple[float, float, float]:
    """
    Compute the shortest perpendicular Euclidean distance from point P(px, py)
    to line segment AB [(ax, ay) -> (bx, by)] in a Cartesian plane.
    Returns:
        (distance_meters, proj_x, proj_y)
    """
    vx = bx - ax
    vy = by - ay
    seg_len_sq = vx * vx + vy * vy

    if seg_len_sq <= 1e-12:
        # Segment is degenerate (single point)
        dx = px - ax
        dy = py - ay
        return np.sqrt(dx * dx + dy * dy), ax, ay

    # Projection scalar t in [0, 1]
    wx = px - ax
    wy = py - ay
    t = (wx * vx + wy * vy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, float(t)))

    proj_x = ax + t_clamped * vx
    proj_y = ay + t_clamped * vy

    dist = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
    return dist, proj_x, proj_y
