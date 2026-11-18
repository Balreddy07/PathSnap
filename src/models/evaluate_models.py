"""
src/models/evaluate_models.py
==============================
Common evaluation framework for all five Phase 3 models.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

GRAPHS_DIR    = None   # set by caller
METRICS_DIR   = None
PRED_DIR      = None
CM_DIR        = None


def set_output_dirs(base: str) -> None:
    global GRAPHS_DIR, METRICS_DIR, PRED_DIR, CM_DIR
    GRAPHS_DIR  = os.path.join(base, "graphs")
    METRICS_DIR = os.path.join(base, "metrics")
    PRED_DIR    = os.path.join(base, "predictions")
    CM_DIR      = os.path.join(base, "confusion_matrices")
    for d in [GRAPHS_DIR, METRICS_DIR, PRED_DIR, CM_DIR]:
        os.makedirs(d, exist_ok=True)


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,       # probability of positive class (or decision scores)
    train_time: float,
    pred_time: float,
    val_df: pd.DataFrame = None,
) -> dict:
    """Compute all evaluation metrics and save artefacts."""

    pos_count = int((y_true == 1).sum())
    neg_count = int((y_true == 0).sum())

    metrics = {
        "model":         model_name,
        "accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":       None,
        "pr_auc":        None,
        "train_time_s":  round(train_time, 2),
        "pred_time_s":   round(pred_time, 4),
        "positives":     pos_count,
        "negatives":     neg_count,
        "total_val":     len(y_true),
    }

    try:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    except Exception as e:
        metrics["roc_auc_error"] = str(e)

    try:
        metrics["pr_auc"] = round(average_precision_score(y_true, y_prob), 4)
    except Exception as e:
        metrics["pr_auc_error"] = str(e)

    print(f"\n  [{model_name}] Results:")
    for k, v in metrics.items():
        if k not in ("model",):
            print(f"    {k}: {v}")

    # Save metrics JSON
    if METRICS_DIR:
        path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved: {path}")

    # Save predictions CSV
    if PRED_DIR and val_df is not None:
        pred_df = val_df[["trip_id", "gps_index", "candidate_arc_id", "binary_target"]].copy()
        pred_df["y_pred"]  = y_pred
        pred_df["y_prob"]  = y_prob
        path = os.path.join(PRED_DIR, f"{model_name}_predictions.csv")
        pred_df.to_csv(path, index=False)
        print(f"  Predictions saved: {path}")

    # Confusion matrix
    _save_confusion_matrix(model_name, y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred, zero_division=0)
    if METRICS_DIR:
        path = os.path.join(METRICS_DIR, f"{model_name}_classification_report.txt")
        with open(path, "w") as f:
            f.write(report)

    return metrics


def _save_confusion_matrix(model_name: str, y_true, y_pred) -> None:
    if not CM_DIR:
        return
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Predicted", ylabel="True",
        title=f"Confusion Matrix – {model_name}",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_roc_curves(all_results: list, y_true: np.ndarray, all_probs: dict) -> None:
    if not GRAPHS_DIR:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in all_probs.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc = roc_auc_score(y_true, probs)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves – All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "roc_curves_all_models.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  ROC curves saved: {path}")


def plot_pr_curves(y_true: np.ndarray, all_probs: dict) -> None:
    if not GRAPHS_DIR:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in all_probs.items():
        try:
            prec, rec, _ = precision_recall_curve(y_true, probs)
            ap = average_precision_score(y_true, probs)
            ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
        except Exception:
            pass
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves – All Models")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "pr_curves_all_models.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  PR curves saved: {path}")


def plot_metric_bars(all_metrics: list, metric_names: list = None) -> None:
    if not GRAPHS_DIR or not all_metrics:
        return
    if metric_names is None:
        metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    model_names = [m["model"] for m in all_metrics]
    x = np.arange(len(model_names))
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        vals = [m.get(metric) or 0 for m in all_metrics]
        bars = ax.bar(x + i * width, vals, width, label=metric)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set(xlabel="Model", ylabel="Score", title="Model Comparison – All Metrics",
           xticks=x + width * (len(metric_names)-1)/2, xticklabels=model_names)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "model_comparison_all_metrics.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"  Comparison chart saved: {path}")

    # Individual bar charts
    for metric in metric_names:
        vals = [m.get(metric) or 0 for m in all_metrics]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bars2 = ax2.bar(model_names, vals, color="steelblue")
        for bar, val in zip(bars2, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax2.set(xlabel="Model", ylabel=metric.upper(), title=f"{metric.upper()} Comparison",
                ylim=(0, 1.15))
        plt.tight_layout()
        p2 = os.path.join(GRAPHS_DIR, f"comparison_{metric}.png")
        fig2.savefig(p2, dpi=100)
        plt.close(fig2)


def plot_timing(all_metrics: list) -> None:
    if not GRAPHS_DIR or not all_metrics:
        return
    model_names = [m["model"] for m in all_metrics]
    train_times = [m.get("train_time_s", 0) for m in all_metrics]
    pred_times  = [m.get("pred_time_s", 0) for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(model_names, train_times, color="coral")
    axes[0].set(title="Training Time (s)", ylabel="Seconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    axes[1].bar(model_names, pred_times, color="steelblue")
    axes[1].set(title="Prediction Time (s)", ylabel="Seconds")

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, "timing_comparison.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def build_comparison_table(all_metrics: list) -> pd.DataFrame:
    rows = []
    for m in all_metrics:
        rows.append({
            "Model":           m["model"],
            "Accuracy":        m.get("accuracy"),
            "Precision":       m.get("precision"),
            "Recall":          m.get("recall"),
            "F1":              m.get("f1"),
            "ROC-AUC":         m.get("roc_auc"),
            "PR-AUC":          m.get("pr_auc"),
            "Train Time (s)":  m.get("train_time_s"),
            "Pred Time (s)":   m.get("pred_time_s"),
        })
    return pd.DataFrame(rows)
