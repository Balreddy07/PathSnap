"""
tests/test_end_to_end.py
========================
End-to-end integration test for the full PathSnap reconstruction pipeline.
Runs candidate generation, ML emissions, HMM Viterbi, and metrics evaluation
on a real sample trip.
"""

import os
import pytest
from src.hmm.route_reconstruction import reconstruct_trip_route

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_full_pipeline_reconstruction():
    trip_id = "00000005"
    raw_root = os.path.join(PROJECT_ROOT, "data", "raw", "train")

    res = reconstruct_trip_route(
        trip_id=trip_id,
        raw_root=raw_root,
        beta=20.0,
        search_radius_m=50.0,
        max_k=10,
    )

    assert res.get("status") != "NO_CANDIDATES"
    assert res["n_observations"] > 0
    assert len(res["reconstructed_route"]) > 0
    assert len(res["ground_truth_route"]) > 0

    metrics = res["metrics_hmm"]
    assert "route_precision" in metrics
    assert "route_recall" in metrics
    assert "route_f1" in metrics
    assert "normalized_edit_distance" in metrics

    assert 0.0 <= metrics["route_f1"] <= 1.0
    assert 0.0 <= metrics["normalized_edit_distance"] <= 1.0
