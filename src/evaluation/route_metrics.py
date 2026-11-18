"""
src/evaluation/route_metrics.py
===============================
Route-level evaluation metrics comparing reconstructed road routes against ground-truth routes.
Includes Baseline 1 (Geometric Nearest) vs Baseline 2 (ML-Only) vs Full PathSnap (HMM/Viterbi).
"""

import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter


def levenshtein_distance(seq1: List[int], seq2: List[int]) -> int:
    """Computes exact edit distance between two sequences of arc IDs."""
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m

    # Optimized two-row dynamic programming
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            )
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def evaluate_route_match(
    predicted_arcs: List[int],
    ground_truth_arcs: List[int],
    arc_lengths: Dict[int, float] = None,
) -> Dict[str, Any]:
    """
    Evaluates reconstructed route sequence against ground truth.

    Returns:
    --------
    dict with:
        - precision: fraction of predicted arcs that are in ground truth
        - recall: fraction of ground truth arcs that are in predicted
        - f1: harmonic mean of precision and recall
        - edit_distance: Levenshtein distance
        - normalized_edit_distance: edit_distance / max(len_pred, len_true)
        - length_error_pct: relative difference in route physical length
        - n_predicted_arcs: int
        - n_ground_truth_arcs: int
    """
    if len(predicted_arcs) == 0 or len(ground_truth_arcs) == 0:
        return {
            "route_precision": 0.0,
            "route_recall": 0.0,
            "route_f1": 0.0,
            "edit_distance": max(len(predicted_arcs), len(ground_truth_arcs)),
            "normalized_edit_distance": 1.0,
            "length_error_pct": 100.0,
            "n_predicted_arcs": len(predicted_arcs),
            "n_ground_truth_arcs": len(ground_truth_arcs),
        }

    # Multiset overlap to account for repeated arcs in loops/traversals
    c_pred = Counter(predicted_arcs)
    c_true = Counter(ground_truth_arcs)

    intersection = sum((c_pred & c_true).values())
    total_pred = len(predicted_arcs)
    total_true = len(ground_truth_arcs)

    precision = intersection / total_pred if total_pred > 0 else 0.0
    recall = intersection / total_true if total_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Edit distance (order-sensitive sequence alignment)
    # Truncate to reasonable length if massive to ensure sub-millisecond calculation
    max_eval_len = 1500
    s_pred = predicted_arcs[:max_eval_len]
    s_true = ground_truth_arcs[:max_eval_len]
    ed = levenshtein_distance(s_pred, s_true)
    norm_ed = ed / max(len(s_pred), len(s_true)) if max(len(s_pred), len(s_true)) > 0 else 0.0

    # Route physical length calculation
    len_error_pct = 0.0
    if arc_lengths:
        pred_len = sum(arc_lengths.get(a, 20.0) for a in predicted_arcs)
        true_len = sum(arc_lengths.get(a, 20.0) for a in ground_truth_arcs)
        if true_len > 0:
            len_error_pct = abs(pred_len - true_len) / true_len * 100.0

    return {
        "route_precision": round(precision, 4),
        "route_recall": round(recall, 4),
        "route_f1": round(f1, 4),
        "edit_distance": int(ed),
        "normalized_edit_distance": round(norm_ed, 4),
        "length_error_pct": round(len_error_pct, 2),
        "n_predicted_arcs": total_pred,
        "n_ground_truth_arcs": total_true,
    }
