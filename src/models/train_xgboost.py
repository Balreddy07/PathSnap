"""
src/models/train_xgboost.py
=============================
XGBoost classifier for PathSnap Phase 3.
Compatible with XGBoost 1.x and 2.x APIs.
"""

import os, sys, time, pickle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

from src.models.data_loader    import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import prepare_all, get_feature_list
from src.models.evaluate_models import evaluate_model, set_output_dirs

MODEL_NAME = "XGBoost"
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "xgboost")


def _compute_scale_pos_weight(y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0


def plot_feature_importance(model, feature_names, graphs_dir):
    scores = model.feature_importances_
    fi = pd.Series(scores, index=feature_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="bar", ax=ax, color="darkorange")
    ax.set(title="XGBoost – Feature Importance (gain)", ylabel="Importance", xlabel="Feature")
    plt.tight_layout()
    path = os.path.join(graphs_dir, "xgb_feature_importance.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Feature importance saved: {path}")
    return fi


def train(config=None):
    if config is None:
        config = load_config()
    base_out = os.path.join(PROJECT_ROOT, "outputs", "phase3")
    set_output_dirs(base_out)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, val_df = load_datasets(config)
    train_df = add_binary_target(train_df)
    val_df   = add_binary_target(val_df)

    cfg      = config["models"]["xgboost"]
    features = get_feature_list(config)

    print(f"\n[{MODEL_NAME}] Preparing features...")
    X_train, X_val, y_train, y_val, imp = prepare_all(train_df, val_df, config, scale=False)

    spw = _compute_scale_pos_weight(y_train)
    print(f"[{MODEL_NAME}] scale_pos_weight = {spw:.2f}")

    xgb_version = tuple(int(x) for x in xgb.__version__.split(".")[:2])

    # Build model – early stopping handled differently in v1 vs v2
    model_kwargs = dict(
        n_estimators    = cfg["n_estimators"],
        max_depth       = cfg["max_depth"],
        learning_rate   = cfg["learning_rate"],
        subsample       = cfg["subsample"],
        colsample_bytree= cfg["colsample_bytree"],
        random_state    = cfg["random_state"],
        eval_metric     = cfg["eval_metric"],
        scale_pos_weight= spw,
        verbosity       = cfg["verbosity"],
        tree_method     = "hist",
        n_jobs          = -1,
        use_label_encoder=False if xgb_version < (1, 6) else None,
    )
    # Remove None kwargs
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    if xgb_version >= (2, 0):
        model_kwargs["early_stopping_rounds"] = cfg.get("early_stopping_rounds", 20)

    model = xgb.XGBClassifier(**model_kwargs)

    print(f"[{MODEL_NAME}] Training on {len(y_train):,} samples (XGBoost {xgb.__version__})...")
    t0 = time.time()

    if xgb_version >= (2, 0):
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=cfg.get("early_stopping_rounds", 20),
                verbose=False,
            )
        except TypeError:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    train_time = time.time() - t0
    print(f"[{MODEL_NAME}] Training complete in {train_time:.1f}s")

    t0 = time.time()
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - t0

    metrics = evaluate_model(MODEL_NAME, y_val, y_pred, y_prob, train_time, pred_time, val_df)

    graphs_dir = os.path.join(base_out, "graphs")
    fi = plot_feature_importance(model, features, graphs_dir)
    fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    # Also save native XGBoost format
    model.save_model(os.path.join(MODEL_DIR, "model.json"))
    print(f"[{MODEL_NAME}] Model saved: {model_path}")

    return metrics, model, imp


if __name__ == "__main__":
    train()
