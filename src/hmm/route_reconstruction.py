"""
src/hmm/route_reconstruction.py
===============================
Complete pipeline for single or multi-trip map-matching and route reconstruction.
Integrates GPS preprocessing, road spatial indexing, candidate generation, ML emissions,
HMM transition modeling, and Viterbi decoding.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.gps_preprocessing import preprocess_gps_trajectory
from src.preprocessing.road_preprocessing import preprocess_road_network
from src.utils.spatial_index import RoadNetworkSpatialIndex
from src.preprocessing.candidate_generation import generate_trip_candidates
from src.hmm.emission_probability import compute_emission_probabilities
from src.hmm.transition_probability import RoadNetworkTransitionModel
from src.hmm.viterbi import run_viterbi
from src.evaluation.route_metrics import evaluate_route_match

FEATURE_COLS = [
    "distance_to_candidate", "gps_speed", "gps_bearing", "road_bearing",
    "bearing_difference", "segment_length", "time_difference",
    "elapsed_time_ratio", "distance_from_start_km", "is_outlier", "is_stationary"
]


def load_raw_trip_data(raw_root: str, trip_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads .track, .nodes, .arcs, .route for a trip."""
    # Ensure zero-padded trip directory if needed
    candidates_to_try = [str(trip_id), str(trip_id).zfill(8)]
    trip_path = None
    actual_tid = trip_id
    for tid in candidates_to_try:
        p = os.path.join(raw_root, tid)
        if os.path.exists(p):
            trip_path = p
            actual_tid = tid
            break

    if trip_path is None:
        raise FileNotFoundError(f"Raw trip directory not found for trip_id {trip_id} in {raw_root}")

    track_path = os.path.join(trip_path, f"{actual_tid}.track")
    nodes_path = os.path.join(trip_path, f"{actual_tid}.nodes")
    arcs_path  = os.path.join(trip_path, f"{actual_tid}.arcs")
    route_path = os.path.join(trip_path, f"{actual_tid}.route")

    # .track: longitude latitude timestamp
    track_df = pd.read_csv(track_path, sep=r"\s+", header=None, names=["longitude", "latitude", "timestamp"])

    nodes_tmp = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["longitude", "latitude"])
    nodes_df  = nodes_tmp.reset_index().rename(columns={"index": "node_id"}).set_index("node_id")

    arcs_tmp = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
    arcs_df  = arcs_tmp.reset_index().rename(columns={"index": "arc_id"}).set_index("arc_id")

    if os.path.exists(route_path):
        route_df = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])
    else:
        route_df = pd.DataFrame(columns=["arc_id"])

    return track_df, nodes_df, arcs_df, route_df


def reconstruct_trip_route(
    trip_id: str,
    raw_root: str = None,
    model: Any = None,
    beta: float = 20.0,
    search_radius_m: float = 50.0,
    max_k: int = 10,
) -> Dict[str, Any]:
    """
    Executes the full PathSnap reconstruction pipeline for a single trip.
    """
    t_start = time.time()
    if raw_root is None:
        raw_root = os.path.join(PROJECT_ROOT, "data", "raw", "train")

    # 1. Load raw files
    track_df, nodes_df, arcs_df, route_df = load_raw_trip_data(raw_root, trip_id)
    ground_truth_arcs = route_df["arc_id"].tolist() if "arc_id" in route_df.columns else []

    # 2. GPS Preprocessing
    gps_df = preprocess_gps_trajectory(track_df, trip_id)

    # 3. Road Network Preprocessing & Graph
    arcs_processed, G = preprocess_road_network(nodes_df, arcs_df)

    # 4. Spatial Index & Candidate Generation
    spatial_index = RoadNetworkSpatialIndex(arcs_processed, nodes_df)
    candidates_df = generate_trip_candidates(
        gps_df, spatial_index, trip_id,
        search_radius_m=search_radius_m, max_k=max_k
    )

    if len(candidates_df) == 0:
        return {
            "trip_id": trip_id,
            "status": "NO_CANDIDATES",
            "n_observations": len(gps_df),
            "reconstructed_route": [],
            "ground_truth_route": ground_truth_arcs,
        }

    # 5. ML Emission Probabilities
    if model is None:
        # Load best model from Phase 4 or Phase 3
        p4_model_path = os.path.join(PROJECT_ROOT, "outputs", "phase4", "models", "xgboost_optimized.pkl")
        p3_model_path = os.path.join(PROJECT_ROOT, "models", "xgboost", "model.pkl")
        chosen_path = p4_model_path if os.path.exists(p4_model_path) else p3_model_path
        with open(chosen_path, "rb") as f:
            model = pickle.load(f)

    candidates_with_emissions = compute_emission_probabilities(
        candidates_df, model, FEATURE_COLS
    )

    # 6. Build Transition Model
    # Prepare arc_info mapping
    arc_info = {}
    arc_lengths = {}
    for aid, row in arcs_processed.iterrows():
        arc_info[int(aid)] = {
            "from_node_id": int(row["from_node_id"]),
            "to_node_id": int(row["to_node_id"]),
            "segment_length": float(row["segment_length"]),
        }
        arc_lengths[int(aid)] = float(row["segment_length"])

    transition_model = RoadNetworkTransitionModel(
        road_graph=G,
        arc_info=arc_info,
        beta=beta,
    )

    # 7. Format inputs for Viterbi
    gps_obs_list = []
    candidates_by_obs = []

    for g_idx, grp in candidates_with_emissions.groupby("gps_index", sort=True):
        first_row = grp.iloc[0]
        gps_obs_list.append({
            "gps_index": int(g_idx),
            "lon": float(first_row["gps_longitude"]),
            "lat": float(first_row["gps_latitude"]),
        })
        c_list = []
        for _, c_row in grp.iterrows():
            c_list.append({
                "candidate_arc_id": int(c_row["candidate_arc_id"]),
                "log_emission_prob": float(c_row["log_emission_prob"]),
                "emission_prob": float(c_row["emission_prob"]),
                "distance_to_candidate": float(c_row["distance_to_candidate"]),
                "raw_ml_prob": float(c_row["raw_ml_prob"]),
            })
        candidates_by_obs.append(c_list)

    # 8. Run Viterbi
    viterbi_res = run_viterbi(gps_obs_list, candidates_by_obs, transition_model)
    reconstructed_arcs = viterbi_res["reconstructed_route_arcs"]

    # 9. Compute Baselines
    # Baseline 1: Geometric Nearest (closest candidate at each observation)
    geo_baseline_arcs = []
    for c_list in candidates_by_obs:
        if c_list:
            best_geo = min(c_list, key=lambda x: x["distance_to_candidate"])
            if len(geo_baseline_arcs) == 0 or geo_baseline_arcs[-1] != best_geo["candidate_arc_id"]:
                geo_baseline_arcs.append(best_geo["candidate_arc_id"])

    # Baseline 2: ML-Only (highest raw ML prob at each observation)
    ml_baseline_arcs = []
    for c_list in candidates_by_obs:
        if c_list:
            best_ml = max(c_list, key=lambda x: x["raw_ml_prob"])
            if len(ml_baseline_arcs) == 0 or ml_baseline_arcs[-1] != best_ml["candidate_arc_id"]:
                ml_baseline_arcs.append(best_ml["candidate_arc_id"])

    # 10. Evaluate all three methods against Ground Truth
    eval_hmm = evaluate_route_match(reconstructed_arcs, ground_truth_arcs, arc_lengths)
    eval_geo = evaluate_route_match(geo_baseline_arcs, ground_truth_arcs, arc_lengths)
    eval_ml  = evaluate_route_match(ml_baseline_arcs, ground_truth_arcs, arc_lengths)

    duration = time.time() - t_start

    return {
        "trip_id": trip_id,
        "duration_s": round(duration, 2),
        "n_observations": len(gps_obs_list),
        "n_candidates_total": len(candidates_df),
        "n_fallbacks": viterbi_res["n_fallbacks"],
        "viterbi_score": viterbi_res["viterbi_score"],
        "reconstructed_route": reconstructed_arcs,
        "geometric_baseline_route": geo_baseline_arcs,
        "ml_baseline_route": ml_baseline_arcs,
        "ground_truth_route": ground_truth_arcs,
        "metrics_hmm": eval_hmm,
        "metrics_geometric_baseline": eval_geo,
        "metrics_ml_baseline": eval_ml,
        "gps_trajectory": track_df[["longitude", "latitude"]].values.tolist(),
        "arcs_processed": arcs_processed,
    }
