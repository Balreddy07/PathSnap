"""
src/models/train_mlp.py
========================
Multi-Layer Perceptron classifier for PathSnap Phase 3.
Uses scikit-learn MLPClassifier with early stopping.
"""

import os, sys, time, pickle
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

from src.models.data_loader    import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import prepare_all, save_scaler
from src.models.evaluate_models import evaluate_model, set_output_dirs

MODEL_NAME = "MLP"
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "mlp")


def _class_weights_to_sample_weights(y, class_weight="balanced"):
    """Convert balanced class weight to sample weights for MLP."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_cls = len(classes)
    weight_map = {c: total / (n_cls * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y])


def train(config=None):
    if config is None:
        config = load_config()
    base_out = os.path.join(PROJECT_ROOT, "outputs", "phase3")
    set_output_dirs(base_out)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df, val_df = load_datasets(config)
    train_df = add_binary_target(train_df)
    val_df   = add_binary_target(val_df)

    cfg = config["models"]["mlp"]

    print(f"\n[{MODEL_NAME}] Preparing features (scaled)...")
    X_train, X_val, y_train, y_val, pipe = prepare_all(train_df, val_df, config, scale=True)

    model = MLPClassifier(
        hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
        activation=cfg["activation"],
        solver=cfg["solver"],
        learning_rate_init=cfg["learning_rate_init"],
        max_iter=cfg["max_iter"],
        early_stopping=cfg["early_stopping"],
        validation_fraction=cfg["validation_fraction"],
        n_iter_no_change=cfg["n_iter_no_change"],
        random_state=cfg["random_state"],
        batch_size=cfg["batch_size"],
        verbose=False,
    )

    print(f"[{MODEL_NAME}] Training on {len(y_train):,} samples...")
    # Use sample weights for class imbalance (MLPClassifier doesn't have class_weight)
    sample_weights = _class_weights_to_sample_weights(y_train)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[{MODEL_NAME}] Training complete in {train_time:.1f}s  "
          f"(iterations: {model.n_iter_})")

    t0 = time.time()
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    pred_time = time.time() - t0

    metrics = evaluate_model(MODEL_NAME, y_val, y_pred, y_prob, train_time, pred_time, val_df)

    # Plot training loss curve
    if hasattr(model, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(model.loss_curve_, label="Training loss")
        if hasattr(model, "validation_scores_") and model.validation_scores_ is not None:
            ax.plot(model.validation_scores_, label="Validation score")
        ax.set(xlabel="Iteration", ylabel="Loss / Score", title="MLP Training History")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(base_out, "graphs", "mlp_training_history.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  Training history saved: {path}")

    model_path  = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    save_scaler(pipe, scaler_path)
    print(f"[{MODEL_NAME}] Model saved: {model_path}")

    return metrics, model, pipe


if __name__ == "__main__":
    train()
