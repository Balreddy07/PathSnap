"""
tests/test_phase4_optimization.py
=================================
Unit tests for Phase 4 Hyperparameter Optimization artifacts and model behavior.
"""

import os
import json
import pickle
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase4")


def test_xgboost_params_exist():
    p = os.path.join(P4_DIR, "best_parameters", "xgboost_best_params.json")
    assert os.path.exists(p), "XGBoost best params file missing"
    with open(p) as f:
        params = json.load(f)
    assert "max_depth" in params
    assert "learning_rate" in params
    assert "n_estimators" in params
    assert params["max_depth"] >= 3


def test_random_forest_params_exist():
    p = os.path.join(P4_DIR, "best_parameters", "random_forest_best_params.json")
    assert os.path.exists(p), "Random Forest best params file missing"
    with open(p) as f:
        params = json.load(f)
    assert "n_estimators" in params
    assert "max_depth" in params


def test_xgboost_model_loading_and_inference():
    p = os.path.join(P4_DIR, "models", "xgboost_optimized.pkl")
    assert os.path.exists(p), "Optimized XGBoost model file missing"
    with open(p, "rb") as f:
        model = pickle.load(f)

    # Synthetic sample of 11 features
    X_dummy = np.random.randn(5, 11)
    probs = model.predict_proba(X_dummy)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_comparison_table_metrics():
    p = os.path.join(P4_DIR, "metrics", "phase4_model_comparison_table.csv")
    assert os.path.exists(p), "Phase 4 model comparison table missing"
    df = pd.read_csv(p)
    assert len(df) >= 4
    assert "Model" in df.columns
    assert "F1" in df.columns
    assert "PR-AUC" in df.columns
    # Ensure all F1 scores are valid numbers above 0.6
    assert (df["F1"] > 0.60).all()
