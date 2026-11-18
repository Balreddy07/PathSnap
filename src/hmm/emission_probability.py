"""
src/hmm/emission_probability.py
===============================
Converts ML candidate prediction scores into normalized HMM emission probabilities.
P(O_t | S_t = a_i)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any


def compute_emission_probabilities(
    candidate_df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    temperature: float = 1.0,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """
    Computes calibrated and normalized emission probabilities for every candidate arc
    belonging to each GPS observation in a trip.

    Parameters:
    -----------
    candidate_df : pd.DataFrame
        DataFrame of candidate arcs with columns: ['trip_id', 'gps_index', 'candidate_arc_id', ...]
    model : trained ML estimator with predict_proba
    feature_cols : list of predictive features
    temperature : float, softmax temperature scaling (default 1.0)
    epsilon : float, smoothing floor to prevent log(0)

    Returns:
    --------
    pd.DataFrame: original DataFrame with new columns:
        - raw_ml_prob: raw probability from binary classifier
        - emission_prob: observation-normalized emission probability
        - log_emission_prob: log(emission_prob) for numerical stability
    """
    df = candidate_df.copy()

    # Prepare features
    X = df[feature_cols].copy()
    for col in ["is_outlier", "is_stationary"]:
        if col in X.columns:
            X[col] = X[col].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Raw model probabilities
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(X.values)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X.values)
        raw_probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        raw_probs = np.ones(len(df)) * 0.5

    df["raw_ml_prob"] = raw_probs

    # Softmax / sum normalization per GPS observation (vectorized)
    temp = max(0.1, temperature)
    scaled_scores = df["raw_ml_prob"] / temp
    max_scores = scaled_scores.groupby([df["trip_id"], df["gps_index"]]).transform("max")
    exp_scores = np.exp(scaled_scores - max_scores)
    sum_exp = exp_scores.groupby([df["trip_id"], df["gps_index"]]).transform("sum")
    norm_probs = (exp_scores / (sum_exp + epsilon)).clip(lower=epsilon, upper=1.0)
    # Re-normalize after clip
    sum_norm = norm_probs.groupby([df["trip_id"], df["gps_index"]]).transform("sum")
    df["emission_prob"] = norm_probs / sum_norm
    df["log_emission_prob"] = np.log(df["emission_prob"])
    return df
