"""
src/visualization/route_map_generator.py
========================================
Interactive Folium and static Matplotlib route visualizations.
Overlays raw GPS trajectory, road network candidates, ground-truth route,
and reconstructed Viterbi route.
"""

import os
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def generate_interactive_route_map(
    trip_result: Dict[str, Any],
    out_html_path: str,
):
    """
    Creates an interactive multi-layer Folium map for the trip.
    Layers:
        - Raw GPS Trajectory (cyan dots / polyline)
        - Ground-Truth Route (green thick polyline)
        - PathSnap Reconstructed Route (blue polyline)
    """
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    gps_coords = trip_result["gps_trajectory"]  # [[lon, lat], ...]
    if not gps_coords:
        return

    # Folium expects [lat, lon]
    folium_gps = [[p[1], p[0]] for p in gps_coords]
    mid_lat = np.mean([p[0] for p in folium_gps])
    mid_lon = np.mean([p[1] for p in folium_gps])

    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Raw GPS Points
    gps_group = folium.FeatureGroup(name="Raw GPS Trajectory")
    folium.PolyLine(folium_gps, color="#00bcd4", weight=2, opacity=0.7, dash_array="5, 5").add_to(gps_group)
    for p in folium_gps[::max(1, len(folium_gps) // 50)]:
        folium.CircleMarker(location=p, radius=3, color="#0097a7", fill=True, fill_opacity=0.8).add_to(gps_group)
    gps_group.add_to(m)

    # Layer 2: Ground-Truth Route
    arcs_processed = trip_result["arcs_processed"]
    gt_arcs = trip_result["ground_truth_route"]
    gt_group = folium.FeatureGroup(name="Ground-Truth Route")
    gt_segments = []
    for aid in gt_arcs:
        if aid in arcs_processed.index:
            row = arcs_processed.loc[aid]
            gt_segments.append([[row["from_lat"], row["from_lon"]], [row["to_lat"], row["to_lon"]]])

    for seg in gt_segments:
        folium.PolyLine(seg, color="#2e7d32", weight=5, opacity=0.8, tooltip="Ground-Truth").add_to(gt_group)
    gt_group.add_to(m)

    # Layer 3: Reconstructed Route
    recon_arcs = trip_result["reconstructed_route"]
    recon_group = folium.FeatureGroup(name="PathSnap Reconstructed Route")
    recon_segments = []
    for aid in recon_arcs:
        if aid in arcs_processed.index:
            row = arcs_processed.loc[aid]
            recon_segments.append([[row["from_lat"], row["from_lon"]], [row["to_lat"], row["to_lon"]]])

    for seg in recon_segments:
        folium.PolyLine(seg, color="#d32f2f", weight=3.5, opacity=0.85, dash_array="6, 3", tooltip="Reconstructed").add_to(recon_group)
    recon_group.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(out_html_path)
    print(f"  Interactive map saved: {out_html_path}")


def generate_static_route_comparison_plot(
    trip_result: Dict[str, Any],
    out_png_path: str,
):
    """
    Generates a 3-panel static matplotlib comparison plot:
    1. Raw GPS Trajectory
    2. Ground Truth Route
    3. Reconstructed Route vs Ground Truth overlay
    """
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    gps_coords = trip_result["gps_trajectory"]
    arcs_processed = trip_result["arcs_processed"]
    gt_arcs = trip_result["ground_truth_route"]
    recon_arcs = trip_result["reconstructed_route"]

    gps_lons = [p[0] for p in gps_coords]
    gps_lats = [p[1] for p in gps_coords]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Panel 1: GPS Points
    axes[0].plot(gps_lons, gps_lats, "c.-", alpha=0.6, markersize=3, label="GPS Trajectory")
    axes[0].set_title(f"Trip {trip_result['trip_id']} – Raw GPS Points", fontsize=12)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Ground-Truth Route
    for aid in gt_arcs:
        if aid in arcs_processed.index:
            row = arcs_processed.loc[aid]
            axes[1].plot([row["from_lon"], row["to_lon"]], [row["from_lat"], row["to_lat"]],
                         color="green", linewidth=2.5, alpha=0.8)
    axes[1].set_title("Ground-Truth Route Sequence", fontsize=12)
    axes[1].set_xlabel("Longitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Overlay (Reconstructed vs Ground Truth)
    for aid in gt_arcs:
        if aid in arcs_processed.index:
            row = arcs_processed.loc[aid]
            axes[2].plot([row["from_lon"], row["to_lon"]], [row["from_lat"], row["to_lat"]],
                         color="green", linewidth=3.5, alpha=0.5, label="Ground-Truth" if aid == gt_arcs[0] else "")
    for aid in recon_arcs:
        if aid in arcs_processed.index:
            row = arcs_processed.loc[aid]
            axes[2].plot([row["from_lon"], row["to_lon"]], [row["from_lat"], row["to_lat"]],
                         color="red", linestyle="--", linewidth=2.0, alpha=0.9, label="PathSnap Viterbi" if aid == recon_arcs[0] else "")

    metrics = trip_result["metrics_hmm"]
    axes[2].set_title(f"Reconstructed Route (F1={metrics['route_f1']:.3f}, EditDist={metrics['normalized_edit_distance']:.3f})", fontsize=12)
    axes[2].set_xlabel("Longitude")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_png_path, dpi=120)
    plt.close(fig)
    print(f"  Static comparison plot saved: {out_png_path}")
