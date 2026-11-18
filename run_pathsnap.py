"""
run_pathsnap.py
===============
End-to-end Map-Matching & Route Reconstruction CLI for PathSnap.

Integrates:
Raw GPS Trajectory + Road Network
  -> GPS Preprocessing
  -> Road Spatial Indexing
  -> Candidate Arc Generation
  -> Feature Extraction
  -> Optimized ML Classifier (Emission Probability)
  -> Road Network Topology (Transition Probability)
  -> HMM / Viterbi Sequence Decoding
  -> Route Infilling & Continuity
  -> Ground-Truth Route Evaluation
  -> Interactive Map & Plot Generation

Usage:
  python run_pathsnap.py --trip-id 00000081
  python run_pathsnap.py --trip-id 00000081 --visualize
  python run_pathsnap.py --all-trips --evaluate
  python run_pathsnap.py --model xgboost --evaluate
"""

import os
import sys
import argparse
import json
import time
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".mplconfig"))

from src.hmm.route_reconstruction import reconstruct_trip_route
from src.visualizations.route_map_generator import (
    generate_interactive_route_map,
    generate_static_route_comparison_plot,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase5")


def get_available_trips(raw_root: str = None) -> list:
    if raw_root is None:
        raw_root = os.path.join(PROJECT_ROOT, "data", "raw", "train")
    if not os.path.exists(raw_root):
        return []
    return sorted([d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))])


def load_best_model(model_type: str = "xgboost"):
    """Loads optimized Phase 4 or baseline Phase 3 model."""
    if model_type.lower() in ("xgboost", "xgb"):
        p4 = os.path.join(PROJECT_ROOT, "outputs", "phase4", "models", "xgboost_optimized.pkl")
        p3 = os.path.join(PROJECT_ROOT, "models", "xgboost", "model.pkl")
    elif model_type.lower() in ("random_forest", "rf"):
        p4 = os.path.join(PROJECT_ROOT, "outputs", "phase4", "models", "random_forest_optimized.pkl")
        p3 = os.path.join(PROJECT_ROOT, "models", "random_forest", "model.pkl")
    else:
        p4 = None
        p3 = os.path.join(PROJECT_ROOT, "models", model_type.lower(), "model.pkl")

    chosen = p4 if (p4 and os.path.exists(p4)) else p3
    if not os.path.exists(chosen):
        raise FileNotFoundError(f"Model file not found: {chosen}. Train model first.")

    print(f"Loading classifier: {chosen}")
    with open(chosen, "rb") as f:
        return pickle.load(f), chosen


def process_trip(
    trip_id: str,
    model: any,
    raw_root: str,
    visualize: bool = False,
    beta: float = 20.0,
) -> dict:
    print(f"\n[{trip_id}] Running PathSnap Reconstruction Pipeline...")
    res = reconstruct_trip_route(trip_id=trip_id, raw_root=raw_root, model=model, beta=beta)

    if res.get("status") == "NO_CANDIDATES":
        print(f"  Warning: No candidates found for trip {trip_id}")
        return res

    m_hmm = res["metrics_hmm"]
    m_geo = res["metrics_geometric_baseline"]
    m_ml  = res["metrics_ml_baseline"]

    print(f"  Observations: {res['n_observations']} | Candidates: {res['n_candidates_total']} | Time: {res['duration_s']}s")
    print(f"  HMM/Viterbi Route:   F1={m_hmm['route_f1']:.4f} | Recall={m_hmm['route_recall']:.4f} | EditDist={m_hmm['normalized_edit_distance']:.4f}")
    print(f"  ML-Only Baseline:    F1={m_ml['route_f1']:.4f}  | Recall={m_ml['route_recall']:.4f} | EditDist={m_ml['normalized_edit_distance']:.4f}")
    print(f"  Geometric Baseline:  F1={m_geo['route_f1']:.4f} | Recall={m_geo['route_recall']:.4f} | EditDist={m_geo['normalized_edit_distance']:.4f}")

    # Save reconstructed route JSON
    route_out_dir = os.path.join(OUT_DIR, "reconstructed_routes")
    os.makedirs(route_out_dir, exist_ok=True)
    out_json = {
        "trip_id": trip_id,
        "n_observations": res["n_observations"],
        "duration_s": res["duration_s"],
        "reconstructed_route_arcs": res["reconstructed_route"],
        "ground_truth_route_arcs": res["ground_truth_route"],
        "metrics_hmm": m_hmm,
        "metrics_ml_baseline": m_ml,
        "metrics_geometric_baseline": m_geo,
    }
    with open(os.path.join(route_out_dir, f"{trip_id}_reconstructed_route.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    # Visualizations
    if visualize:
        vis_dir = os.path.join(OUT_DIR, "visualizations")
        html_path = os.path.join(vis_dir, f"{trip_id}_interactive_route_map.html")
        png_path  = os.path.join(vis_dir, f"{trip_id}_route_comparison.png")
        generate_interactive_route_map(res, html_path)
        generate_static_route_comparison_plot(res, png_path)

    return res


def main():
    parser = argparse.ArgumentParser(description="PathSnap Map-Matching & Route Reconstruction")
    parser.add_argument("--trip-id", type=str, default=None, help="Specific trip ID to reconstruct (e.g. 00000081)")
    parser.add_argument("--all-trips", action="store_true", help="Process all available validation trips")
    parser.add_argument("--evaluate", action="store_true", help="Run comparative route evaluation across trips")
    parser.add_argument("--model", type=str, default="xgboost", help="Classifier model type: xgboost or random_forest")
    parser.add_argument("--visualize", action="store_true", help="Generate interactive Folium HTML and static PNG plots")
    parser.add_argument("--beta", type=float, default=20.0, help="HMM transition beta parameter (meters)")
    parser.add_argument("--limit", type=int, default=10, help="Max trips to process when using --all-trips")
    args = parser.parse_args()

    raw_root = os.path.join(PROJECT_ROOT, "data", "raw", "train")
    model, model_path = load_best_model(args.model)

    trips_to_run = []
    if args.trip_id:
        trips_to_run = [args.trip_id]
    elif args.all_trips or args.evaluate:
        # Load validation trip IDs from val.csv or directory
        val_csv_path = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path, usecols=["trip_id"])
            val_trips = sorted(val_df["trip_id"].unique())
            # Format as string with leading zeros if numeric
            trips_to_run = [f"{int(t):08d}" if str(t).isdigit() else str(t) for t in val_trips]
        else:
            trips_to_run = get_available_trips(raw_root)
        if args.limit:
            trips_to_run = trips_to_run[:args.limit]
    else:
        # Default: pick first trip
        all_avail = get_available_trips(raw_root)
        trips_to_run = all_avail[:1] if all_avail else []

    print(f"Running PathSnap on {len(trips_to_run)} trip(s): {trips_to_run}")

    all_results = []
    for tid in trips_to_run:
        res = process_trip(tid, model, raw_root, visualize=args.visualize, beta=args.beta)
        if res.get("status") != "NO_CANDIDATES":
            all_results.append(res)

    # Comparative evaluation table across trips
    if len(all_results) > 0 and (args.evaluate or args.all_trips or len(all_results) > 1):
        summary_rows = []
        for r in all_results:
            tid = r["trip_id"]
            h = r["metrics_hmm"]
            g = r["metrics_geometric_baseline"]
            m = r["metrics_ml_baseline"]
            summary_rows.append({
                "trip_id": tid,
                "hmm_f1": h["route_f1"],
                "hmm_recall": h["route_recall"],
                "hmm_norm_edit_dist": h["normalized_edit_distance"],
                "hmm_len_err_pct": h["length_error_pct"],
                "ml_baseline_f1": m["route_f1"],
                "ml_norm_edit_dist": m["normalized_edit_distance"],
                "geo_baseline_f1": g["route_f1"],
                "geo_norm_edit_dist": g["normalized_edit_distance"],
            })

        summary_df = pd.DataFrame(summary_rows)
        metrics_out_dir = os.path.join(OUT_DIR, "route_metrics")
        os.makedirs(metrics_out_dir, exist_ok=True)
        summary_csv = os.path.join(metrics_out_dir, "route_evaluation_comparison.csv")
        summary_df.to_csv(summary_csv, index=False)

        print("\n" + "=" * 80)
        print("ROUTE RECONSTRUCTION COMPARATIVE EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Evaluated Trips: {len(summary_df)}")
        print(f"  Average HMM/Viterbi Route F1:        {summary_df['hmm_f1'].mean():.4f}")
        print(f"  Average HMM/Viterbi Recall:          {summary_df['hmm_recall'].mean():.4f}")
        print(f"  Average HMM/Viterbi Norm Edit Dist:  {summary_df['hmm_norm_edit_dist'].mean():.4f}")
        print(f"  Average ML-Only Baseline F1:         {summary_df['ml_baseline_f1'].mean():.4f}")
        print(f"  Average ML-Only Norm Edit Dist:      {summary_df['ml_norm_edit_dist'].mean():.4f}")
        print(f"  Average Geometric Baseline F1:       {summary_df['geo_baseline_f1'].mean():.4f}")
        print(f"  Average Geometric Norm Edit Dist:    {summary_df['geo_norm_edit_dist'].mean():.4f}")
        print(f"\nDetailed metrics saved: {summary_csv}")


if __name__ == "__main__":
    main()
