"""
src/models/data_loader.py
=========================
Shared data loading and validation for all Phase 3 models.
READ-ONLY: never modifies the source CSVs.
"""

import os
import time
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "phase3_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_datasets(config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val CSVs. Returns (train_df, val_df)."""
    if config is None:
        config = load_config()

    train_path = os.path.join(PROJECT_ROOT, config["data"]["train_csv"])
    val_path   = os.path.join(PROJECT_ROOT, config["data"]["val_csv"])

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"val.csv not found: {val_path}")

    print(f"Loading train: {train_path}")
    t0 = time.time()
    train_df = pd.read_csv(train_path)
    print(f"  → {len(train_df):,} rows, {len(train_df.columns)} cols  ({time.time()-t0:.1f}s)")

    print(f"Loading val:   {val_path}")
    t0 = time.time()
    val_df = pd.read_csv(val_path)
    print(f"  → {len(val_df):,} rows, {len(val_df.columns)} cols  ({time.time()-t0:.1f}s)")

    return train_df, val_df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive binary_target = (candidate_arc_id == true_arc_id).astype(int).
    Does NOT modify the original label column.
    """
    df = df.copy()
    df["binary_target"] = (df["candidate_arc_id"] == df["true_arc_id"]).astype(int)
    return df


def apply_distance_filter(df: pd.DataFrame, threshold_m: float) -> Tuple[pd.DataFrame, int]:
    """
    Optionally filter rows where distance_to_true_arc > threshold_m.
    Returns (filtered_df, n_excluded).
    """
    if threshold_m is None:
        return df, 0
    mask = df["distance_to_true_arc"].notna() & (df["distance_to_true_arc"] <= threshold_m)
    excluded = (~mask).sum()
    return df[mask].copy(), int(excluded)


def validate_split(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate trip-level split integrity."""
    train_trips = set(train_df["trip_id"].unique())
    val_trips   = set(val_df["trip_id"].unique())
    overlap     = train_trips & val_trips
    return {
        "n_train_trips": len(train_trips),
        "n_val_trips":   len(val_trips),
        "train_trip_ids": sorted(train_trips),
        "val_trip_ids":   sorted(val_trips),
        "overlap":        sorted(overlap),
        "overlap_count":  len(overlap),
        "split_valid":    len(overlap) == 0,
    }


def validate_target(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute target statistics and coverage."""
    results = {}
    for name, df in [("train", train_df), ("val", val_df)]:
        pos  = int((df["binary_target"] == 1).sum())
        neg  = int((df["binary_target"] == 0).sum())
        total = len(df)
        # Coverage: GPS obs that have at least one positive candidate
        grp = df.groupby(["trip_id", "gps_index"])
        total_obs    = grp.ngroups
        obs_with_pos = df[df["binary_target"] == 1].groupby(["trip_id", "gps_index"]).ngroups
        dist_col = df["distance_to_true_arc"]
        results[name] = {
            "total_rows": total,
            "positives":  pos,
            "negatives":  neg,
            "pct_pos":    round(100 * pos / total, 2) if total > 0 else 0,
            "pct_neg":    round(100 * neg / total, 2) if total > 0 else 0,
            "total_gps_obs":    total_obs,
            "obs_with_pos":     obs_with_pos,
            "coverage_pct":     round(100 * obs_with_pos / total_obs, 2) if total_obs > 0 else 0,
            "dist_mean_m":      round(dist_col.mean(), 2) if dist_col.notna().any() else None,
            "dist_median_m":    round(dist_col.median(), 2) if dist_col.notna().any() else None,
            "dist_pct_gt_10m":  round(100 * (dist_col > 10).mean(), 2),
            "dist_pct_gt_20m":  round(100 * (dist_col > 20).mean(), 2),
            "dist_pct_gt_50m":  round(100 * (dist_col > 50).mean(), 2),
        }
    return results


def dataset_audit(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Full audit: load, add target, validate split and target."""
    if config is None:
        config = load_config()

    train_df, val_df = load_datasets(config)
    train_df = add_binary_target(train_df)
    val_df   = add_binary_target(val_df)

    split_info  = validate_split(train_df, val_df)
    target_info = validate_target(train_df, val_df)

    # Null checks on feature columns
    feat_cols = config["features"]["predictive"]
    train_nulls = {c: int(train_df[c].isna().sum()) for c in feat_cols if c in train_df.columns}
    val_nulls   = {c: int(val_df[c].isna().sum())   for c in feat_cols if c in val_df.columns}

    return {
        "split":       split_info,
        "target":      target_info,
        "train_nulls": {k: v for k, v in train_nulls.items() if v > 0},
        "val_nulls":   {k: v for k, v in val_nulls.items()   if v > 0},
    }


if __name__ == "__main__":
    import json
    cfg   = load_config()
    audit = dataset_audit(cfg)
    print(json.dumps(audit, indent=2, default=str))
