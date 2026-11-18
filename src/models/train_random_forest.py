"""
src/models/train_random_forest.py
==================================
Random Forest classifier for PathSnap Phase 3.
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
from sklearn.ensemble import RandomForestClassifier

from src.models.data_loader    import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import prepare_all, get_feature_list
from src.models.evaluate_models import evaluate_model, set_output_dirs, GRAPHS_DIR

MODEL_NAME = "RandomForest"
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "random_forest")


def plot_feature_importance(model, feature_names, graphs_dir):
    fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    fi.plot(kind="bar", ax=ax, color="steelblue")
    ax.set(title="Random Forest – Feature Importance", ylabel="Importance", xlabel="Feature")
    plt.tight_layout()
    path = os.path.join(graphs_dir, "rf_feature_importance.png")
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

    cfg = config["models"]["random_forest"]
    features = get_feature_list(config)

    print(f"\n[{MODEL_NAME}] Preparing features (no scaling)...")
    X_train, X_val, y_train, y_val, imp = prepare_all(train_df, val_df, config, scale=False)

    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
    )

    print(f"[{MODEL_NAME}] Training on {len(y_train):,} samples...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[{MODEL_NAME}] Training complete in {train_time:.1f}s")

    t0 = time.time()
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - t0

    metrics = evaluate_model(MODEL_NAME, y_val, y_pred, y_prob, train_time, pred_time, val_df)

    # Feature importance
    graphs_dir = os.path.join(base_out, "graphs")
    fi = plot_feature_importance(model, features, graphs_dir)
    fi_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    fi.to_csv(fi_path)
    print(f"  Feature importance CSV: {fi_path}")

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[{MODEL_NAME}] Model saved: {model_path}")

    return metrics, model, imp


if __name__ == "__main__":
    train()
