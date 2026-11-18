"""tests/test_phase3_evaluation.py"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import pytest, numpy as np, json
from sklearn.metrics import f1_score

PHASE3_DIR  = os.path.join(PROJECT_ROOT, "outputs", "phase3")
MODEL_NAMES = ["LogisticRegression", "SVM_Linear", "RandomForest", "XGBoost", "MLP"]

def test_metrics_files_exist():
    metrics_dir = os.path.join(PHASE3_DIR, "metrics")
    missing = []
    for name in MODEL_NAMES:
        path = os.path.join(metrics_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            missing.append(name)
    assert len(missing) == 0, f"Missing metrics for: {missing}"

def test_metrics_values_in_range():
    metrics_dir = os.path.join(PHASE3_DIR, "metrics")
    for name in MODEL_NAMES:
        path = os.path.join(metrics_dir, f"{name}_metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        for metric in ["accuracy", "precision", "recall", "f1"]:
            val = m.get(metric)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{name}.{metric}={val} out of range"

def test_predictions_files_exist():
    pred_dir = os.path.join(PHASE3_DIR, "predictions")
    missing = []
    for name in MODEL_NAMES:
        path = os.path.join(pred_dir, f"{name}_predictions.csv")
        if not os.path.exists(path):
            missing.append(name)
    assert len(missing) == 0, f"Missing predictions for: {missing}"

def test_predictions_consistent_with_metrics():
    import pandas as pd
    pred_dir    = os.path.join(PHASE3_DIR, "predictions")
    metrics_dir = os.path.join(PHASE3_DIR, "metrics")
    for name in MODEL_NAMES:
        p_path = os.path.join(pred_dir,    f"{name}_predictions.csv")
        m_path = os.path.join(metrics_dir, f"{name}_metrics.json")
        if not os.path.exists(p_path) or not os.path.exists(m_path):
            continue
        preds = pd.read_csv(p_path)
        with open(m_path) as f:
            metrics = json.load(f)
        y_true = preds["binary_target"].values
        y_pred = preds["y_pred"].values
        f1 = f1_score(y_true, y_pred, zero_division=0)
        saved_f1 = metrics.get("f1", 0)
        assert abs(f1 - saved_f1) < 0.01, f"{name}: f1 mismatch {f1:.4f} vs {saved_f1:.4f}"
