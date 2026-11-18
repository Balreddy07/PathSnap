"""
src/optimization/optimize_random_forest.py
==========================================
Hyperparameter optimization for Random Forest using Optuna and GroupKFold cross-validation.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.data_loader import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import prepare_X_y, get_feature_list
from src.optimization.optimization_utils import (
    load_phase4_config, get_group_cv, evaluate_trial_predictions, save_optimization_plots
)
from src.models.evaluate_models import evaluate_model, set_output_dirs

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_optimization(n_trials: int = 12):
    print("=" * 60)
    print("Phase 4: Hyperparameter Optimization – Random Forest")
    print("=" * 60)

    p3_cfg = load_config()
    p4_cfg = load_phase4_config()
    features = get_feature_list(p3_cfg)

    train_df, val_df = load_datasets(p3_cfg)
    train_df = add_binary_target(train_df)
    val_df   = add_binary_target(val_df)

    X_train_df, y_train = prepare_X_y(train_df, features)
    X_val_df, y_val = prepare_X_y(val_df, features)

    X_train = X_train_df.values
    y_train_vals = y_train.values
    X_val = X_val_df.values
    y_val_vals = y_val.values

    # For efficient search across trials, sample 16 representative trips
    unique_trips = train_df["trip_id"].unique()
    np.random.seed(42)
    sample_trips = np.random.choice(unique_trips, size=16, replace=False)
    search_mask = train_df["trip_id"].isin(sample_trips)
    search_df = train_df[search_mask].copy()

    X_search = X_train[search_mask]
    y_search = y_train_vals[search_mask]

    splits = get_group_cv(search_df, n_splits=2, group_col="trip_id")
    print(f"Random Forest search dataset: {len(search_df):,} samples across 16 trips, 2 folds.")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 160, step=40),
            "max_depth": trial.suggest_int("max_depth", 12, 22, step=2),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

        fold_scores = []
        for tr_idx, va_idx in splits:
            rf = RandomForestClassifier(**params)
            rf.fit(X_search[tr_idx], y_search[tr_idx])
            preds_prob = rf.predict_proba(X_search[va_idx])[:, 1]
            metrics = evaluate_trial_predictions(y_search[va_idx], preds_prob)
            fold_scores.append(metrics["f1"])

        return np.mean(fold_scores)

    print(f"Starting Optuna search ({n_trials} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_duration = time.time() - t0
    print(f"Optimization finished in {opt_duration:.1f}s. Best CV F1: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    # Retrain final model on full training set with 200 estimators
    best_params = study.best_params.copy()
    best_params["n_estimators"] = 200  # full ensemble
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    print("\nTraining final optimized Random Forest on full dataset (1,034,891 samples)...")
    best_model = RandomForestClassifier(**best_params)
    t_train = time.time()
    best_model.fit(X_train, y_train_vals)
    final_train_time = time.time() - t_train

    # Evaluate on final validation set
    t_pred = time.time()
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    final_pred_time = time.time() - t_pred

    out_base = os.path.join(PROJECT_ROOT, "outputs", "phase4")
    set_output_dirs(out_base)
    val_metrics = evaluate_model(
        "RandomForest_Optimized", y_val_vals, val_preds, val_probs,
        final_train_time, final_pred_time, val_df
    )

    # Save artifacts
    best_params_path = os.path.join(out_base, "best_parameters", "random_forest_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    model_dir = os.path.join(out_base, "models")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "random_forest_optimized.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)

    history_df = study.trials_dataframe()
    history_path = os.path.join(out_base, "optimization_history", "random_forest_optimization_history.csv")
    history_df.to_csv(history_path, index=False)
    save_optimization_plots(study, "RandomForest", os.path.join(out_base, "plots"))

    print(f"\nRandom Forest Optimization Completed:")
    print(f"  Validation F1: {val_metrics['f1']:.4f} (PR-AUC: {val_metrics['pr_auc']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f})")
    print(f"  Artifacts saved to {out_base}")

    return val_metrics, best_model, study.best_params


if __name__ == "__main__":
    run_optimization()
