"""
src/trajectory_analysis.py
==========================
Deep trajectory analysis for PathSnap GPS observations (.track):
- Rigorous geodesic displacement calculation (Haversine formula on WGS84 sphere)
- High-precision timestamp interval delta-t distribution (verifies ~1 Hz nominal sampling)
- Pointwise velocity computation (m/s and km/h)
- Detection and characterization of GPS anomalies (dropout gaps, teleportation jumps, stationary drift)
- Aggregation across all 80 training trips (and student_test)
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.data_loader import (
    get_available_partitions,
    get_trip_ids,
    load_trip,
)
from src.data_validation import haversine_vectorized


def extract_pointwise_trajectory_metrics(partition: str = "train") -> Dict[str, np.ndarray]:
    """
    Extract pointwise arrays across all trips in a partition:
    - all_dt: consecutive timestamp intervals (seconds)
    - all_displacements: consecutive point displacements (meters)
    - all_speeds_mps: consecutive point speeds (m/s)
    - all_speeds_kmh: consecutive point speeds (km/h)
    - all_lons: longitude coordinates
    - all_lats: latitude coordinates
    """
    trip_ids = get_trip_ids(partition)

    dt_list = []
    disp_list = []
    speed_list = []
    lon_list = []
    lat_list = []

    for trip_id in trip_ids:
        trip = load_trip(trip_id, partition)
        track = trip["track"]
        if len(track) < 2:
            continue

        timestamps = track["timestamp"].values
        dt = np.diff(timestamps)
        dt_list.append(dt)

        lons = track["longitude"].values
        lats = track["latitude"].values
        lon_list.append(lons)
        lat_list.append(lats)

        disps = haversine_vectorized(lons[:-1], lats[:-1], lons[1:], lats[1:])
        disp_list.append(disps)

        safe_dt = np.where(dt > 0, dt, 1e-6)
        speeds = disps / safe_dt
        speed_list.append(speeds)

    return {
        "all_dt": np.concatenate(dt_list) if dt_list else np.array([]),
        "all_displacements": np.concatenate(disp_list) if disp_list else np.array([]),
        "all_speeds_mps": np.concatenate(speed_list) if speed_list else np.array([]),
        "all_speeds_kmh": np.concatenate(speed_list) * 3.6 if speed_list else np.array([]),
        "all_lons": np.concatenate(lon_list) if lon_list else np.array([]),
        "all_lats": np.concatenate(lat_list) if lat_list else np.array([]),
    }


def compute_trajectory_summary_statistics(partition: str = "train") -> Dict[str, Any]:
    """Calculate aggregate summary statistics for trajectories in the partition."""
    metrics = extract_pointwise_trajectory_metrics(partition)

    dts = metrics["all_dt"]
    disps = metrics["all_displacements"]
    speeds_kmh = metrics["all_speeds_kmh"]

    pct_1s = float(np.mean(np.isclose(dts, 1.0, atol=1e-3)) * 100.0) if len(dts) > 0 else 0.0
    pct_sub_2s = float(np.mean(dts <= 2.0) * 100.0) if len(dts) > 0 else 0.0
    large_gaps = int(np.sum(dts > 5.0))
    zero_gaps = int(np.sum(dts == 0.0))
    negative_gaps = int(np.sum(dts < 0.0))

    high_speed_anomalies = int(np.sum(speeds_kmh > 150.0))
    stationary_points = int(np.sum(disps < 0.5))

    return {
        "total_observations": int(len(metrics["all_lons"])),
        "total_consecutive_steps": int(len(dts)),
        "dt_mean_seconds": float(np.mean(dts)) if len(dts) > 0 else 0.0,
        "dt_median_seconds": float(np.median(dts)) if len(dts) > 0 else 0.0,
        "dt_min_seconds": float(np.min(dts)) if len(dts) > 0 else 0.0,
        "dt_max_seconds": float(np.max(dts)) if len(dts) > 0 else 0.0,
        "pct_exact_1s": pct_1s,
        "pct_sub_2s": pct_sub_2s,
        "large_time_gaps_gt_5s": large_gaps,
        "duplicate_timestamps": zero_gaps,
        "non_monotonic_timestamps": negative_gaps,
        "displacement_mean_meters": float(np.mean(disps)) if len(disps) > 0 else 0.0,
        "displacement_median_meters": float(np.median(disps)) if len(disps) > 0 else 0.0,
        "displacement_max_meters": float(np.max(disps)) if len(disps) > 0 else 0.0,
        "speed_mean_kmh": float(np.mean(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0,
        "speed_median_kmh": float(np.median(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0,
        "speed_max_kmh": float(np.max(speeds_kmh)) if len(speeds_kmh) > 0 else 0.0,
        "high_speed_anomalies_gt_150kmh": high_speed_anomalies,
        "stationary_drift_points_lt_0_5m": stationary_points,
    }


if __name__ == "__main__":
    summary = compute_trajectory_summary_statistics("train")
    print("Trajectory Summary (train):")
    for k, v in summary.items():
        print(f"  {k}: {v}")
