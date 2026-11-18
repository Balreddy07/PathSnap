"""
src/hmm/viterbi.py
==================
Viterbi dynamic programming algorithm for HMM map-matching in log-probability space.
Reconstructs the optimal chronological road-arc sequence and infills intermediate graph paths.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from src.hmm.transition_probability import RoadNetworkTransitionModel


def run_viterbi(
    gps_observations: List[Dict[str, Any]],
    candidates_by_obs: List[List[Dict[str, Any]]],
    transition_model: RoadNetworkTransitionModel,
) -> Dict[str, Any]:
    """
    Executes Viterbi decoding over the candidate state lattice.

    Parameters:
    -----------
    gps_observations : List[Dict]
        List of dicts per observation, e.g. [{'gps_index': 0, 'lon': x, 'lat': y}, ...]
    candidates_by_obs : List[List[Dict]]
        For each observation t, a list of candidate dicts:
        [{'candidate_arc_id': id, 'log_emission_prob': val, 'distance_to_candidate': dist, ...}, ...]
    transition_model : RoadNetworkTransitionModel

    Returns:
    --------
    Dict containing:
        - matched_candidate_arcs: List[int], the selected candidate arc for each GPS observation
        - reconstructed_route_arcs: List[int], continuous sequence of traversed road arcs
        - viterbi_score: float, optimal cumulative log probability
        - n_observations: int
        - n_fallbacks: int, count of disconnected transition penalties encountered
        - status: str, 'SUCCESS' or 'FALLBACK'
    """
    T = len(gps_observations)
    if T == 0:
        return {
            "matched_candidate_arcs": [],
            "reconstructed_route_arcs": [],
            "viterbi_score": -np.inf,
            "n_observations": 0,
            "n_fallbacks": 0,
            "status": "EMPTY",
        }

    # Step 1: Initialize Viterbi DP table and backpointers
    # V[t] = array of best log-probs for candidates at time t
    V: List[np.ndarray] = []
    backpointers: List[List[int]] = []
    transition_paths: List[List[List[int]]] = []  # path_arcs[t][i] = traversed arcs from ptr to cand i
    fallback_flags: List[List[bool]] = []

    # t = 0: Initial state distribution
    t0_cands = candidates_by_obs[0]
    if len(t0_cands) == 0:
        # Edge case: No candidates found for first point
        V.append(np.array([-50.0]))
        backpointers.append([-1])
        transition_paths.append([[]])
        fallback_flags.append([True])
    else:
        init_v = np.array([float(c.get("log_emission_prob", -1.0)) for c in t0_cands])
        V.append(init_v)
        backpointers.append([-1] * len(t0_cands))
        transition_paths.append([[c["candidate_arc_id"]] for c in t0_cands])
        fallback_flags.append([False] * len(t0_cands))

    # Step 2: Forward Recursion
    for t in range(1, T):
        prev_cands = candidates_by_obs[t - 1]
        curr_cands = candidates_by_obs[t]
        prev_pt = (gps_observations[t - 1]["lon"], gps_observations[t - 1]["lat"])
        curr_pt = (gps_observations[t]["lon"], gps_observations[t]["lat"])

        n_curr = len(curr_cands)
        n_prev = len(prev_cands)

        if n_curr == 0:
            # Observation without candidates: propagate best previous score with penalty
            best_prev_idx = int(np.argmax(V[t - 1]))
            V.append(np.array([V[t - 1][best_prev_idx] - 20.0]))
            backpointers.append([best_prev_idx])
            transition_paths.append([[]])
            fallback_flags.append([True])
            continue

        curr_v = np.zeros(n_curr)
        curr_ptr = [-1] * n_curr
        curr_paths = [[] for _ in range(n_curr)]
        curr_fallbacks = [False] * n_curr

        for i, c_curr in enumerate(curr_cands):
            c_curr_id = c_curr["candidate_arc_id"]
            em_log_p = float(c_curr.get("log_emission_prob", -1.0))

            best_val = -np.inf
            best_j = 0
            best_path = [c_curr_id]
            is_fallback = False

            for j, c_prev in enumerate(prev_cands):
                c_prev_id = c_prev["candidate_arc_id"]
                log_trans_p, path_arcs = transition_model.compute_transition(
                    prev_arc_id=c_prev_id,
                    curr_arc_id=c_curr_id,
                    prev_gps_point=prev_pt,
                    curr_gps_point=curr_pt,
                )

                candidate_val = V[t - 1][j] + log_trans_p + em_log_p

                if candidate_val > best_val:
                    best_val = candidate_val
                    best_j = j
                    best_path = path_arcs
                    is_fallback = (log_trans_p <= getattr(transition_model, "penalty_log_prob", -50.0))

            curr_v[i] = best_val
            curr_ptr[i] = best_j
            curr_paths[i] = best_path
            curr_fallbacks[i] = is_fallback

        V.append(curr_v)
        backpointers.append(curr_ptr)
        transition_paths.append(curr_paths)
        fallback_flags.append(curr_fallbacks)

    # Step 3: Backtracking
    # Find candidate with maximum likelihood at final time step T-1
    best_last_idx = int(np.argmax(V[T - 1]))
    total_score = float(V[T - 1][best_last_idx])

    # Trace back indices
    chosen_indices = [0] * T
    chosen_indices[T - 1] = best_last_idx
    for t in range(T - 1, 0, -1):
        chosen_indices[t - 1] = backpointers[t][chosen_indices[t]]

    # Step 4: Extract matched candidates and reconstruct continuous route
    matched_candidates = []
    reconstructed_route = []
    total_fallbacks = 0

    for t in range(T):
        cands_at_t = candidates_by_obs[t]
        idx = chosen_indices[t]

        if idx >= 0 and idx < len(cands_at_t):
            matched_candidates.append(cands_at_t[idx]["candidate_arc_id"])
        else:
            matched_candidates.append(-1)

        if fallback_flags[t][idx]:
            total_fallbacks += 1

        # Collect segment arcs from transition
        step_arcs = transition_paths[t][idx]
        for arc_id in step_arcs:
            if len(reconstructed_route) == 0 or reconstructed_route[-1] != arc_id:
                reconstructed_route.append(arc_id)

    return {
        "matched_candidate_arcs": matched_candidates,
        "reconstructed_route_arcs": reconstructed_route,
        "viterbi_score": total_score,
        "n_observations": T,
        "n_fallbacks": total_fallbacks,
        "status": "SUCCESS" if total_fallbacks < (T * 0.5) else "FALLBACK",
    }
