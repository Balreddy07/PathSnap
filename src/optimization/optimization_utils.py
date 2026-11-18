"""
src/optimization/optimization_utils.py
======================================
Utilities for GroupKFold cross-validation, Optuna logging, and visualization.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_phase4_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(PROJECT_ROOT, "configs", "phase4_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_group_cv(df: pd.DataFrame, n_splits: int = 3, group_col: str = "trip_id"):
    """
    Generate GroupKFold splits so that no trip_id appears in both train and val folds.
    """
    groups = df[group_col].values
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=groups))


def evaluate_trial_predictions(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def save_optimization_plots(study, model_name: str, out_dir: str):
    """Save optimization history and parameter importance plots."""
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Optimization history
    trials = [t for t in study.trials if t.value is not None]
    if trials:
        trial_numbers = [t.number for t in trials]
        trial_values = [t.value for t in trials]
        best_values = np.maximum.accumulate(trial_values)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trial_numbers, trial_values, "o-", alpha=0.6, label="Trial F1")
        ax.plot(trial_numbers, best_values, "r-", linewidth=2, label="Best F1 so far")
        ax.set(xlabel="Trial", ylabel="Validation F1", title=f"{model_name} – Optimization History")
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(out_dir, f"{model_name.lower()}_optimization_history.png")
        fig.savefig(hist_path, dpi=100)
        plt.close(fig)

    # 2. Parameter importances if optuna visualization is available
    try:
        import optuna.importance
        importances = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(8, 5))
        params = list(importances.keys())
        scores = list(importances.values())
        y_pos = np.arange(len(params))
        ax.barh(y_pos, scores, align="center", color="teal")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set(xlabel="Relative Importance", title=f"{model_name} – Hyperparameter Importance")
        plt.tight_layout()
        imp_path = os.path.join(out_dir, f"{model_name.lower()}_param_importance.png")
        fig.savefig(imp_path, dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"Could not generate param importance plot for {model_name}: {e}")
