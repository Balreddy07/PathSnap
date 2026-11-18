"""
src/optimization/optimize_xgboost.py
====================================
Hyperparameter optimization for XGBoost using Optuna and GroupKFold cross-validation.
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb

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


def run_optimization(n_trials: int = 25):
    print("=" * 60)
    print("Phase 4: Hyperparameter Optimization – XGBoost")
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

    # Pre-compute GroupKFold splits on training set
    splits = get_group_cv(train_df, n_splits=p4_cfg["cv"]["n_splits"], group_col="trip_id")
    print(f"GroupKFold: {len(splits)} folds across {train_df['trip_id'].nunique()} training trips.")

    # Positive class weight
    neg = (y_train_vals == 0).sum()
    pos = (y_train_vals == 1).sum()
    spw = float(neg / pos) if pos > 0 else 1.0

    # Subsample folds for fast trial exploration (use 2 folds during search)
    search_splits = splits[:2]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.65, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "scale_pos_weight": spw,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "logloss",
        }

        fold_scores = []
        for tr_idx, va_idx in search_splits:
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train[tr_idx], y_train_vals[tr_idx], verbose=False)
            preds_prob = clf.predict_proba(X_train[va_idx])[:, 1]
            metrics = evaluate_trial_predictions(y_train_vals[va_idx], preds_prob)
            fold_scores.append(metrics["f1"])

        return np.mean(fold_scores)

    print(f"Starting Optuna search ({n_trials} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_duration = time.time() - t0
    print(f"Optimization finished in {opt_duration:.1f}s. Best CV F1: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    # Retrain best model on complete training dataset (1,034,891 samples)
    best_params = study.best_params.copy()
    best_params.update({
        "scale_pos_weight": spw,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "logloss",
    })

    print("\nTraining final optimized XGBoost on full training dataset...")
    best_model = xgb.XGBClassifier(**best_params)
    t_train = time.time()
    best_model.fit(X_train, y_train_vals, verbose=False)
    final_train_time = time.time() - t_train

    # Evaluate on final held-out validation set
    t_pred = time.time()
    val_probs = best_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    final_pred_time = time.time() - t_pred

    out_base = os.path.join(PROJECT_ROOT, "outputs", "phase4")
    set_output_dirs(out_base)
    val_metrics = evaluate_model(
        "XGBoost_Optimized", y_val_vals, val_preds, val_probs,
        final_train_time, final_pred_time, val_df
    )

    # Save artifacts
    best_params_path = os.path.join(out_base, "best_parameters", "xgboost_best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)

    model_dir = os.path.join(out_base, "models")
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "xgboost_optimized.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    best_model.save_model(os.path.join(model_dir, "xgboost_optimized.json"))

    # Save history and plots
    history_df = study.trials_dataframe()
    history_path = os.path.join(out_base, "optimization_history", "xgboost_optimization_history.csv")
    history_df.to_csv(history_path, index=False)
    save_optimization_plots(study, "XGBoost", os.path.join(out_base, "plots"))

    print(f"\nXGBoost Optimization Completed:")
    print(f"  Validation F1: {val_metrics['f1']:.4f} (PR-AUC: {val_metrics['pr_auc']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f})")
    print(f"  Artifacts saved to {out_base}")

    return val_metrics, best_model, study.best_params


if __name__ == "__main__":
    run_optimization()
