"""
tests/test_phase5_hmm.py
========================
Unit tests for Phase 5 HMM / Viterbi route reconstruction components.
"""

import os
import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.hmm.emission_probability import compute_emission_probabilities
from src.hmm.transition_probability import RoadNetworkTransitionModel
from src.hmm.viterbi import run_viterbi
from src.evaluation.route_metrics import (
    levenshtein_distance,
    evaluate_route_match,
)


def test_emission_probability_normalization():
    # Synthetic candidate dataframe for 2 observations
    data = {
        "trip_id": ["00000001"] * 6,
        "gps_index": [0, 0, 0, 1, 1, 1],
        "candidate_arc_id": [10, 11, 12, 20, 21, 22],
        "distance_to_candidate": [5.0, 15.0, 30.0, 2.0, 8.0, 25.0],
        "gps_speed": [10.0] * 6,
        "gps_bearing": [90.0] * 6,
        "road_bearing": [90.0] * 6,
        "bearing_difference": [0.0] * 6,
        "segment_length": [50.0] * 6,
        "time_difference": [1.0] * 6,
        "elapsed_time_ratio": [0.1] * 6,
        "distance_from_start_km": [0.1] * 6,
        "is_outlier": [0] * 6,
        "is_stationary": [0] * 6,
    }
    cand_df = pd.DataFrame(data)

    # Mock model
    class MockModel:
        def predict_proba(self, X):
            # Higher score for smaller distance
            dists = X[:, 0]
            p = 1.0 / (1.0 + dists / 10.0)
            return np.column_stack([1.0 - p, p])

    features = [
        "distance_to_candidate", "gps_speed", "gps_bearing", "road_bearing",
        "bearing_difference", "segment_length", "time_difference",
        "elapsed_time_ratio", "distance_from_start_km", "is_outlier", "is_stationary"
    ]
    res_df = compute_emission_probabilities(cand_df, MockModel(), features)

    assert "emission_prob" in res_df.columns
    assert "log_emission_prob" in res_df.columns

    # Sum of emission probabilities per gps observation must equal 1.0 (approx)
    for _, grp in res_df.groupby("gps_index"):
        assert np.isclose(grp["emission_prob"].sum(), 1.0, atol=1e-4)


def test_transition_probability():
    # Build simple graph: 1 -> 2 -> 3
    G = nx.DiGraph()
    G.add_edge(1, 2, weight=100.0, arc_id=101)
    G.add_edge(2, 3, weight=100.0, arc_id=102)
    G.add_edge(4, 5, weight=100.0, arc_id=104)  # disconnected component

    arc_info = {
        101: {"from_node_id": 1, "to_node_id": 2, "segment_length": 100.0},
        102: {"from_node_id": 2, "to_node_id": 3, "segment_length": 100.0},
        104: {"from_node_id": 4, "to_node_id": 5, "segment_length": 100.0},
    }

    tm = RoadNetworkTransitionModel(G, arc_info, beta=20.0)

    # Connected transition 101 -> 102
    log_p_conn, path_arcs = tm.compute_transition(
        prev_arc_id=101, curr_arc_id=102,
        prev_gps_point=(0.0, 0.0), curr_gps_point=(0.001, 0.0)
    )
    assert log_p_conn > -20.0, f"Expected reasonable log prob, got {log_p_conn}"
    assert 102 in path_arcs

    # Disconnected transition 101 -> 104
    log_p_disc, _ = tm.compute_transition(
        prev_arc_id=101, curr_arc_id=104,
        prev_gps_point=(0.0, 0.0), curr_gps_point=(0.001, 0.0)
    )
    assert log_p_disc <= tm.penalty_log_prob, "Disconnected transition should receive penalty"


def test_viterbi_toy_lattice():
    # 2 observations, each with 2 candidate arcs
    gps_obs = [
        {"gps_index": 0, "lon": 0.0, "lat": 0.0},
        {"gps_index": 1, "lon": 0.001, "lat": 0.0},
    ]

    cands_by_obs = [
        [
            {"candidate_arc_id": 101, "log_emission_prob": -0.2, "emission_prob": 0.8},
            {"candidate_arc_id": 999, "log_emission_prob": -2.0, "emission_prob": 0.2},
        ],
        [
            {"candidate_arc_id": 102, "log_emission_prob": -0.2, "emission_prob": 0.8},
            {"candidate_arc_id": 888, "log_emission_prob": -2.0, "emission_prob": 0.2},
        ],
    ]

    # Simple mock transition model
    class MockTM:
        def compute_transition(self, prev_arc_id, curr_arc_id, prev_gps_point, curr_gps_point):
            if (prev_arc_id, curr_arc_id) == (101, 102):
                return -0.1, [102]
            return -50.0, [curr_arc_id]

    res = run_viterbi(gps_obs, cands_by_obs, MockTM())
    assert res["reconstructed_route_arcs"] == [101, 102]


def test_route_metrics():
    # Exact match
    r1 = [10, 11, 12]
    r_gt = [10, 11, 12]
    m1 = evaluate_route_match(r1, r_gt)
    assert m1["route_precision"] == 1.0
    assert m1["route_recall"] == 1.0
    assert m1["route_f1"] == 1.0
    assert m1["edit_distance"] == 0
    assert m1["normalized_edit_distance"] == 0.0

    # Partial match
    r2 = [10, 99, 12]
    m2 = evaluate_route_match(r2, r_gt)
    assert m2["route_precision"] == pytest.approx(2/3, 0.01)
    assert m2["normalized_edit_distance"] > 0.0
