"""
src/models/train_logistic_regression.py
========================================
Logistic Regression classifier for PathSnap Phase 3.
"""

import os, sys, time, pickle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.data_loader    import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import prepare_all, save_scaler
from src.models.evaluate_models import evaluate_model, set_output_dirs

MODEL_NAME = "LogisticRegression"
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "logistic_regression")


def train(config=None):
    if config is None:
        config = load_config()
    set_output_dirs(os.path.join(PROJECT_ROOT, "outputs", "phase3"))
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    train_df, val_df = load_datasets(config)
    train_df = add_binary_target(train_df)
    val_df   = add_binary_target(val_df)

    cfg = config["models"]["logistic_regression"]

    # Prepare features (scaled)
    print(f"\n[{MODEL_NAME}] Preparing features...")
    X_train, X_val, y_train, y_val, pipe = prepare_all(train_df, val_df, config, scale=True)

    # Train
    print(f"[{MODEL_NAME}] Training...")
    model = LogisticRegression(
        C=cfg["C"],
        max_iter=cfg["max_iter"],
        solver=cfg["solver"],
        class_weight=cfg["class_weight"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[{MODEL_NAME}] Training complete in {train_time:.1f}s")

    # Predict
    t0 = time.time()
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - t0

    # Evaluate
    metrics = evaluate_model(MODEL_NAME, y_val, y_pred, y_prob, train_time, pred_time, val_df)

    # Save model and scaler
    model_path  = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    save_scaler(pipe, scaler_path)
    print(f"[{MODEL_NAME}] Model saved: {model_path}")

    return metrics, model, pipe


if __name__ == "__main__":
    train()
