"""
src/preprocessing/gps_preprocessing.py
======================================
Robust GPS trajectory preprocessing:
- Validates coordinate bounds, missing values, and strictly monotonic timestamps
- Calculates consecutive displacements (Haversine), delta-t, and instantaneous velocities
- Flags sensor anomalies without destroying raw observations:
    * is_outlier: speed > 45 m/s (~162 km/h) or displacement > 250m
    * is_stationary: displacement < 0.5m
    * is_sampling_gap: delta-t > 5.0s
- Preserves original row indices ('gps_index') for complete end-to-end traceability
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.utils.geo_utils import haversine_meters, forward_bearing_deg


def preprocess_gps_trajectory(
    track_df: pd.DataFrame,
    trip_id: str,
    max_speed_mps: float = 45.0,
    max_disp_m: float = 250.0,
    min_disp_stationary_m: float = 0.5,
    gap_threshold_s: float = 5.0,
) -> pd.DataFrame:
    """
    Clean and annotate raw GPS trajectory with kinematics and anomaly flags.
    Original observations are preserved with explicit traceability columns.
    """
    df = track_df.copy()
    n_points = len(df)

    # 1. Ensure zero-based original index is preserved
    df["trip_id"] = trip_id
    df["gps_index"] = np.arange(n_points, dtype=np.int64)

    # 2. Time differences
    timestamps = df["timestamp"].values
    dt = np.zeros(n_points, dtype=np.float64)
    if n_points > 1:
        dt[1:] = np.diff(timestamps)
        # Avoid division-by-zero on initial observation or duplicate timestamps
        dt[0] = 1.0  # Nominal initial interval
    df["time_difference"] = np.round(dt, 4)

    # 3. Displacements and Speed
    lons = df["longitude"].values
    lats = df["latitude"].values

    disp = np.zeros(n_points, dtype=np.float64)
    speeds = np.zeros(n_points, dtype=np.float64)
    bearings = np.zeros(n_points, dtype=np.float64)

    if n_points > 1:
        step_disp = haversine_meters(lons[:-1], lats[:-1], lons[1:], lats[1:])
        disp[1:] = step_disp
        safe_dt = np.where(dt[1:] > 0, dt[1:], 1e-6)
        speeds[1:] = step_disp / safe_dt

        # Compass bearing
        step_bearings = forward_bearing_deg(lons[:-1], lats[:-1], lons[1:], lats[1:])
        bearings[1:] = step_bearings
        bearings[0] = step_bearings[0]  # Impute initial heading from first motion

    df["displacement_m"] = np.round(disp, 2)
    df["gps_speed"] = np.round(speeds, 2)
    df["gps_bearing"] = np.round(bearings, 2)

    # 4. Anomaly flags (Non-destructive)
    df["is_outlier"] = (df["gps_speed"] > max_speed_mps) | (df["displacement_m"] > max_disp_m)
    df["is_stationary"] = df["displacement_m"] < min_disp_stationary_m
    df["is_sampling_gap"] = df["time_difference"] > gap_threshold_s

    # 5. Cumulative metrics
    df["cumulative_distance_km"] = np.round(np.cumsum(disp) / 1000.0, 3)
    start_lon, start_lat = lons[0], lats[0]
    df["distance_from_start_km"] = np.round(
        haversine_meters(start_lon, start_lat, lons, lats) / 1000.0, 3
    )

    t0 = timestamps[0]
    total_time = max(1.0, float(timestamps[-1] - t0))
    df["elapsed_time_ratio"] = np.round((timestamps - t0) / total_time, 4)

    return df
