"""
src/models/compare_models.py
==============================
Loads all saved metrics and generates comparison report and charts.
"""

import os, sys, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.evaluate_models import (
    set_output_dirs, plot_metric_bars, plot_timing, build_comparison_table,
    plot_roc_curves, plot_pr_curves,
)

PHASE3_DIR  = os.path.join(PROJECT_ROOT, "outputs", "phase3")
METRICS_DIR = os.path.join(PHASE3_DIR, "metrics")
GRAPHS_DIR  = os.path.join(PHASE3_DIR, "graphs")
PRED_DIR    = os.path.join(PHASE3_DIR, "predictions")

MODEL_NAMES = ["LogisticRegression", "SVM_Linear", "RandomForest", "XGBoost", "MLP"]


def load_all_metrics():
    set_output_dirs(PHASE3_DIR)
    all_metrics = []
    for name in MODEL_NAMES:
        path = os.path.join(METRICS_DIR, f"{name}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics.append(json.load(f))
            print(f"  Loaded metrics: {name}")
        else:
            print(f"  WARNING: metrics not found for {name}")
    return all_metrics


def load_all_probs(val_df, y_val):
    """Load saved prediction CSVs and extract probability columns."""
    pred_dir = os.path.join(PHASE3_DIR, "predictions")
    all_probs = {}
    for name in MODEL_NAMES:
        path = os.path.join(pred_dir, f"{name}_predictions.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "y_prob" in df.columns:
                all_probs[name] = df["y_prob"].values
    return all_probs


def identify_best_model(all_metrics: list) -> dict:
    """Select best model by F1 score (most informative for imbalanced data)."""
    if not all_metrics:
        return {}
    scored = [(m.get("f1") or 0, m) for m in all_metrics]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_model = scored[0]
    return {"best_model": best_model["model"], "best_f1": best_score, "ranking": scored}


def run_comparison():
    set_output_dirs(PHASE3_DIR)
    all_metrics = load_all_metrics()

    if not all_metrics:
        print("No metrics found. Run training first.")
        return

    # Comparison charts
    plot_metric_bars(all_metrics)
    plot_timing(all_metrics)

    # ROC and PR curves from saved predictions
    pred_dir = os.path.join(PHASE3_DIR, "predictions")
    all_probs = {}
    y_true = None
    for name in MODEL_NAMES:
        path = os.path.join(pred_dir, f"{name}_predictions.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "y_prob" in df.columns:
                all_probs[name] = df["y_prob"].values
                if y_true is None and "binary_target" in df.columns:
                    y_true = df["binary_target"].values

    if y_true is not None and all_probs:
        plot_roc_curves(all_metrics, y_true, all_probs)
        plot_pr_curves(y_true, all_probs)

    # Comparison table
    table = build_comparison_table(all_metrics)
    table_path = os.path.join(PHASE3_DIR, "metrics", "model_comparison_table.csv")
    table.to_csv(table_path, index=False)
    print(f"\nComparison table saved: {table_path}")
    print(table.to_string(index=False))

    # Best model
    best = identify_best_model(all_metrics)
    if best:
        print(f"\n{'='*50}")
        print(f"Best model: {best['best_model']}  (F1 = {best['best_f1']:.4f})")
        print(f"{'='*50}")

    return all_metrics, table, best


if __name__ == "__main__":
    run_comparison()
