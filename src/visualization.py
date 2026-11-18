"""
src/visualization.py
====================
Generates comprehensive visualization artifacts for PathSnap Phase 1:
1. Publication-quality distribution plots (outputs/graphs/*.png)
2. Interactive Folium HTML maps (outputs/maps/*.html)

Adheres to:
- Folium coordinate order: [latitude, longitude]
- True reference route reconstruction using .route and .arcs endpoints
- Distinguished layer colors and legends (Blue = GPS, Green = Reference Route, Gray = Network)
- Layer controls and zoom bounding
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import folium
from folium import plugins

from src.data_loader import (
    load_trip,
    get_trip_ids,
)
from src.trajectory_analysis import extract_pointwise_trajectory_metrics
from src.data_validation import haversine_vectorized


# Configure Seaborn style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15,
})


def generate_eda_plots(
    trip_stats_df: pd.DataFrame,
    road_stats_df: pd.DataFrame,
    route_stats_df: pd.DataFrame,
    output_dir: Path
) -> List[Path]:
    """Generate all 8 required static EDA plots and save to outputs/graphs/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # 1. GPS Observation Counts Distribution
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(
        data=trip_stats_df,
        x="n_gps_observations",
        hue="partition",
        kde=True,
        bins=30,
        ax=ax,
        palette={"train": "#1f77b4", "student_test": "#ff7f0e"}
    )
    ax.set_title("Distribution of GPS Observation Counts per Trip")
    ax.set_xlabel("Number of GPS Observations")
    ax.set_ylabel("Trip Count")
    fig.tight_layout()
    p1 = output_dir / "01_gps_observation_counts.png"
    fig.savefig(p1)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Trajectory Duration Distribution
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.histplot(
        data=trip_stats_df,
        x="duration_minutes",
        hue="partition",
        kde=True,
        bins=25,
        ax=ax,
        palette={"train": "#2ca02c", "student_test": "#d62728"}
    )
    ax.set_title("Distribution of Trajectory Durations")
    ax.set_xlabel("Duration (Minutes)")
    ax.set_ylabel("Trip Count")
    fig.tight_layout()
    p2 = output_dir / "02_trajectory_duration_distribution.png"
    fig.savefig(p2)
    plt.close(fig)
    generated_files.append(p2)

    # Pointwise metrics for 3, 4, 5
    pointwise = extract_pointwise_trajectory_metrics("train")
    dts = pointwise["all_dt"]
    disps = pointwise["all_displacements"]
    speeds_kmh = pointwise["all_speeds_kmh"]
    lons = pointwise["all_lons"]
    lats = pointwise["all_lats"]

    # 3. Timestamp Interval Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    # Clip dt for visualization of nominal vs outlier gaps
    dts_clipped = np.clip(dts, 0.5, 4.5)
    sns.histplot(dts_clipped, bins=40, kde=False, color="#17becf", ax=ax1)
    ax1.set_title(r"Timestamp Interval ($\Delta t$) Distribution (Zoomed 0.5s - 4.5s)")
    ax1.set_xlabel(r"Interval $\Delta t$ (seconds)")
    ax1.set_ylabel("Count (Observations)")
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label=f"1.0s Mode ({np.mean(np.isclose(dts, 1.0, atol=1e-3))*100:.1f}%)")
    ax1.legend()

    # Log-scaled full range
    sns.histplot(np.log10(np.clip(dts, 0.9, 1000)), bins=35, kde=True, color="#9467bd", ax=ax2)
    ax2.set_title(r"Log-Scale Interval Distribution ($\log_{10} \Delta t$)")
    ax2.set_xlabel(r"$\log_{10}(\Delta t \text{ [s]})$")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    p3 = output_dir / "03_timestamp_interval_distribution.png"
    fig.savefig(p3)
    plt.close(fig)
    generated_files.append(p3)

    # 4. GPS Displacement and Speed Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    sns.histplot(np.clip(disps, 0, 50), bins=35, kde=True, color="#8c564b", ax=ax1)
    ax1.set_title("Consecutive GPS Displacement Distribution")
    ax1.set_xlabel("Displacement (meters)")
    ax1.set_ylabel("Count")

    sns.histplot(np.clip(speeds_kmh, 0, 140), bins=35, kde=True, color="#e377c2", ax=ax2)
    ax2.set_title("Pointwise Estimated Speed Distribution")
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Count")
    fig.tight_layout()
    p4 = output_dir / "04_gps_displacement_distribution.png"
    fig.savefig(p4)
    plt.close(fig)
    generated_files.append(p4)

    # 5. GPS Coordinate Distributions (Spatial Density)
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    hb = ax.hexbin(lons, lats, gridsize=50, cmap="YlOrRd", mincnt=1, bins="log")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label(r"$\log_{10}(\text{Observation Count})$")
    ax.set_title("Geographical Distribution of GPS Observations (Darmstadt Area)")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    p5 = output_dir / "05_gps_coordinate_distributions.png"
    fig.savefig(p5)
    plt.close(fig)
    generated_files.append(p5)

    # 6. GPS Anomaly Summaries
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    anomaly_categories = ["Total Obs", "Nominal 1s", "Gaps > 5s", "Speed > 150 km/h", "Drift < 0.5m"]
    counts = [
        len(dts),
        int(np.sum(np.isclose(dts, 1.0, atol=1e-3))),
        int(np.sum(dts > 5.0)),
        int(np.sum(speeds_kmh > 150.0)),
        int(np.sum(disps < 0.5)),
    ]
    ax1.bar(anomaly_categories[1:], counts[1:], color=["#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"])
    ax1.set_title("Observation-Level Anomaly Counts (Train Partition)")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=25)

    # Trip-level max speeds
    sns.boxplot(data=trip_stats_df, x="partition", y="max_speed_kmh", hue="partition", legend=False, ax=ax2, palette="Set2")
    ax2.set_title("Trip-Level Maximum Observed Speed (km/h)")
    ax2.set_xlabel("Partition")
    ax2.set_ylabel("Max Speed (km/h)")
    fig.tight_layout()
    p6 = output_dir / "06_gps_anomaly_summaries.png"
    fig.savefig(p6)
    plt.close(fig)
    generated_files.append(p6)

    # 7. Road Network Degree Distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    sns.histplot(road_stats_df["in_degree_mean"], bins=20, kde=True, color="#1f77b4", ax=ax1)
    ax1.set_title("Mean In-Degree across Trip Road Networks")
    ax1.set_xlabel("Mean In-Degree")
    ax1.set_ylabel("Trip Count")

    sns.scatterplot(
        data=road_stats_df,
        x="n_nodes",
        y="n_arcs",
        hue="partition",
        palette="tab10",
        alpha=0.8,
        ax=ax2
    )
    ax2.set_title("Road Network Scale: Arcs vs Nodes")
    ax2.set_xlabel("Number of Nodes")
    ax2.set_ylabel("Number of Directed Arcs")
    fig.tight_layout()
    p7 = output_dir / "07_road_network_degree_distributions.png"
    fig.savefig(p7)
    plt.close(fig)
    generated_files.append(p7)

    # 8. Route Continuity & Length vs GPS Observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    sns.scatterplot(
        data=route_stats_df,
        x="n_gps_observations",
        y="route_arc_count",
        hue="partition",
        palette="Dark2",
        alpha=0.8,
        ax=ax1
    )
    ax1.set_title("Route Arc Count vs GPS Point Count")
    ax1.set_xlabel("Number of GPS Observations")
    ax1.set_ylabel("Number of Route Arcs")

    sns.histplot(route_stats_df["continuity_pct"], bins=20, kde=False, color="#2ca02c", ax=ax2)
    ax2.set_title("Reference Route Continuity Percentage")
    ax2.set_xlabel("Continuity (%)")
    ax2.set_ylabel("Trip Count")
    fig.tight_layout()
    p8 = output_dir / "08_route_continuity_and_length.png"
    fig.savefig(p8)
    plt.close(fig)
    generated_files.append(p8)

    print(f"[Plots] Generated 8 EDA figures in {output_dir}")
    return generated_files


def generate_interactive_folium_maps(trip_id: str, partition: str, output_dir: Path) -> List[Path]:
    """
    Generate the 3 required interactive Folium maps for a representative trip:
    - MAP 1: Raw GPS trajectory
    - MAP 2: Road network visualization
    - MAP 3: GPS observations with true reference route overlay
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trip = load_trip(trip_id, partition)

    track_df = trip["track"]
    nodes_df = trip["nodes"]
    arcs_df = trip["arcs"]
    route_s = trip["route"]

    # Folium center coordinates: [lat, lon]
    center_lat = float(track_df["latitude"].mean())
    center_lon = float(track_df["longitude"].mean())

    generated_maps = []

    # ==========================================
    # MAP 1: Raw GPS Trajectory
    # ==========================================
    m1 = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (1 Hz)")

    gps_coords = []
    for idx, row in track_df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        t = row["timestamp"]
        gps_coords.append([lat, lon])

        # Add point circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#0066cc",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.7,
            popup=f"Obs #{idx}<br>Time: {t:.1f}s<br>Lat: {lat:.5f}<br>Lon: {lon:.5f}",
        ).add_to(fg_gps)

    # GPS Polyline trace
    folium.PolyLine(
        gps_coords,
        color="#004488",
        weight=2,
        opacity=0.6,
        tooltip=f"Raw Trajectory (Trip {trip_id})",
    ).add_to(fg_gps)

    # Start and End Markers
    folium.Marker(
        location=gps_coords[0],
        popup=f"Start (t={track_df.iloc[0]['timestamp']:.1f}s)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fg_gps)

    folium.Marker(
        location=gps_coords[-1],
        popup=f"End (t={track_df.iloc[-1]['timestamp']:.1f}s)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fg_gps)

    fg_gps.add_to(m1)
    folium.LayerControl().add_to(m1)

    p_map1 = output_dir / "map1_raw_gps_trajectory.html"
    m1.save(str(p_map1))
    generated_maps.append(p_map1)

    # ==========================================
    # MAP 2: Road Network Visualisation
    # ==========================================
    # To keep map HTML lightweight and responsive, select road network arcs within bounding box of trip + margin
    lat_min, lat_max = track_df["latitude"].min() - 0.01, track_df["latitude"].max() + 0.01
    lon_min, lon_max = track_df["longitude"].min() - 0.01, track_df["longitude"].max() + 0.01

    m2 = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    fg_network = folium.FeatureGroup(name="Physical Road Network Segments")

    from_nodes = arcs_df["from_node_id"].values
    to_nodes = arcs_df["to_node_id"].values
    node_lons = nodes_df["longitude"].values
    node_lats = nodes_df["latitude"].values

    # Filter arcs within bounding box
    u_lats = node_lats[from_nodes]
    u_lons = node_lons[from_nodes]
    v_lats = node_lats[to_nodes]
    v_lons = node_lons[to_nodes]

    mask = (
        (u_lats >= lat_min) & (u_lats <= lat_max) & (u_lons >= lon_min) & (u_lons <= lon_max)
    )
    relevant_arc_indices = np.where(mask)[0]

    # Draw sampled road segments
    step = max(1, len(relevant_arc_indices) // 3000)  # Up to 3000 arcs for high performance
    for arc_idx in relevant_arc_indices[::step]:
        u = from_nodes[arc_idx]
        v = to_nodes[arc_idx]
        folium.PolyLine(
            [[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
            color="#555555",
            weight=1.5,
            opacity=0.5,
            tooltip=f"Arc ID: {arc_idx} ({u}->{v})",
        ).add_to(fg_network)

    fg_network.add_to(m2)
    folium.LayerControl().add_to(m2)

    p_map2 = output_dir / "map2_road_network.html"
    m2.save(str(p_map2))
    generated_maps.append(p_map2)

    # ==========================================
    # MAP 3: GPS Observations + Ground-Truth Route Overlay
    # ==========================================
    m3 = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer A: Local road network
    fg_net_overlay = folium.FeatureGroup(name="Road Network Segments (Gray)", show=True)
    for arc_idx in relevant_arc_indices[::step]:
        u = from_nodes[arc_idx]
        v = to_nodes[arc_idx]
        folium.PolyLine(
            [[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]],
            color="#888888",
            weight=1.2,
            opacity=0.4,
        ).add_to(fg_net_overlay)
    fg_net_overlay.add_to(m3)

    # Layer B: Reference Route (Reconstructed from .route and .arcs)
    fg_route = folium.FeatureGroup(name="True Reference Route (Green)", show=True)
    r_arcs = route_s.values
    valid_mask = (r_arcs >= 0) & (r_arcs < len(arcs_df))
    valid_route_arcs = r_arcs[valid_mask]

    route_coords = []
    for arc_id in valid_route_arcs:
        u = from_nodes[arc_id]
        v = to_nodes[arc_id]
        segment = [[node_lats[u], node_lons[u]], [node_lats[v], node_lons[v]]]
        folium.PolyLine(
            segment,
            color="#00aa00",
            weight=4,
            opacity=0.85,
            tooltip=f"Route Arc ID: {arc_id}",
        ).add_to(fg_route)

    fg_route.add_to(m3)

    # Layer C: Raw GPS observations
    fg_gps_overlay = folium.FeatureGroup(name="Noisy GPS Observations (Blue)", show=True)
    for idx, row in track_df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="#0055cc",
            fill=True,
            fill_color="#3388ff",
            fill_opacity=0.7,
            popup=f"GPS #{idx} (t={row['timestamp']:.1f}s)",
        ).add_to(fg_gps_overlay)

    # Start and End markers
    folium.Marker(
        location=[track_df.iloc[0]["latitude"], track_df.iloc[0]["longitude"]],
        popup=f"GPS Start: Trip {trip_id}",
        icon=folium.Icon(color="darkgreen", icon="play"),
    ).add_to(fg_gps_overlay)

    folium.Marker(
        location=[track_df.iloc[-1]["latitude"], track_df.iloc[-1]["longitude"]],
        popup=f"GPS End: Trip {trip_id}",
        icon=folium.Icon(color="darkred", icon="stop"),
    ).add_to(fg_gps_overlay)

    fg_gps_overlay.add_to(m3)

    # Add HTML Legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 220px; height: 110px; 
                border:2px solid #999; z-index:9999; font-size:12px;
                background-color:white; opacity:0.92; padding: 10px;
                border-radius: 6px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
      <b>PathSnap Ground Truth Legend</b><br>
      <i style="background: #3388ff; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 6px;"></i> Noisy GPS Observations<br>
      <i style="background: #00aa00; width: 18px; height: 4px; display: inline-block; margin-right: 6px;"></i> Reference Route (Ground Truth)<br>
      <i style="background: #888888; width: 18px; height: 2px; display: inline-block; margin-right: 6px;"></i> Road Network Segments
    </div>
    '''
    m3.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m3)

    p_map3 = output_dir / "map3_gps_with_reference_route.html"
    m3.save(str(p_map3))
    generated_maps.append(p_map3)

    print(f"[Folium] Generated 3 interactive HTML maps in {output_dir}")
    return generated_maps


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    csv_dir = base_dir / "outputs" / "csv"
    graphs_dir = base_dir / "outputs" / "graphs"
    maps_dir = base_dir / "outputs" / "maps"

    trip_stats = pd.read_csv(csv_dir / "trip_statistics.csv")
    road_stats = pd.read_csv(csv_dir / "road_network_statistics.csv")
    route_stats = pd.read_csv(csv_dir / "route_statistics.csv")

    generate_eda_plots(trip_stats, road_stats, route_stats, graphs_dir)

    # Representative trip: Trip 00000001 (standard urban trajectory with rich network)
    generate_interactive_folium_maps("00000001", "train", maps_dir)
