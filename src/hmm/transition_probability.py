"""
src/hmm/transition_probability.py
=================================
Road-network topology transition probabilities using directed graph shortest paths.
Based on Newson & Krumm (2009) map-matching transition model.
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional
from src.utils.geo_utils import haversine_meters


class RoadNetworkTransitionModel:
    """
    Computes spatial-temporal transition probabilities between candidate road arcs
    using shortest path distance over the directed road network.
    """

    def __init__(
        self,
        road_graph: nx.DiGraph,
        arc_info: Dict[int, Dict[str, float]],
        beta: float = 20.0,
        max_search_dist_m: float = 1200.0,
        penalty_log_prob: float = -50.0,
    ):
        """
        Parameters:
        -----------
        road_graph : nx.DiGraph
            Directed graph where nodes are road intersections and edges are directed arcs.
            Edge attributes include 'arc_id' and 'weight' (meters).
        arc_info : dict
            Mapping arc_id -> {'from_node_id': int, 'to_node_id': int, 'segment_length': float}
        beta : float
            Transition parameter (meters) controlling tolerance to deviation between
            network distance and GPS displacement. Default: 20.0m.
        max_search_dist_m : float
            Cutoff distance for shortest-path Dijkstra search.
        penalty_log_prob : float
            Log probability penalty assigned to disconnected / unreachable transitions.
        """
        self.graph = road_graph
        self.arc_info = arc_info
        self.beta = max(1.0, beta)
        self.max_search_dist_m = max_search_dist_m
        self.penalty_log_prob = penalty_log_prob
        self.log_beta = np.log(self.beta)
        self._path_cache: Dict[Tuple[int, int], Tuple[float, List[int]]] = {}

    def get_shortest_path_info(self, from_node: int, to_node: int) -> Tuple[float, List[int]]:
        """
        Computes (distance, intermediate_nodes) from from_node to to_node.
        Uses bidirectional Dijkstra and in-memory cache for fast transitions.
        """
        if from_node == to_node:
            return 0.0, [from_node]

        cache_key = (from_node, to_node)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Geometric pruning check: Euclidean distance is a strict lower bound
        n_from = self.graph.nodes.get(from_node)
        n_to   = self.graph.nodes.get(to_node)
        if n_from and n_to and "lon" in n_from and "lon" in n_to:
            straight_dist = haversine_meters(n_from["lon"], n_from["lat"], n_to["lon"], n_to["lat"])
            if straight_dist > self.max_search_dist_m:
                res = (float("inf"), [])
                self._path_cache[cache_key] = res
                return res

        try:
            length, path_nodes = nx.bidirectional_dijkstra(
                self.graph, source=from_node, target=to_node,
                weight=lambda u, v, d: d.get("length", 10.0)
            )
            if length > self.max_search_dist_m:
                res = (float("inf"), [])
            else:
                res = (float(length), path_nodes)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            res = (float("inf"), [])

        self._path_cache[cache_key] = res
        return res

    def compute_transition(
        self,
        prev_arc_id: int,
        curr_arc_id: int,
        prev_gps_point: Tuple[float, float],
        curr_gps_point: Tuple[float, float],
    ) -> Tuple[float, List[int]]:
        """
        Computes log P(curr_arc | prev_arc) and list of intermediate road arc IDs.

        Parameters:
        -----------
        prev_arc_id : int
        curr_arc_id : int
        prev_gps_point : (lon, lat)
        curr_gps_point : (lon, lat)

        Returns:
        --------
        (log_prob, intermediate_arc_ids)
        """
        d_gps = haversine_meters(
            prev_gps_point[0], prev_gps_point[1],
            curr_gps_point[0], curr_gps_point[1]
        )

        # Retrieve arc topology
        prev_info = self.arc_info.get(prev_arc_id)
        curr_info = self.arc_info.get(curr_arc_id)

        if prev_info is None or curr_info is None:
            return self.penalty_log_prob, []

        u1, v1 = prev_info["from_node_id"], prev_info["to_node_id"]
        u2, v2 = curr_info["from_node_id"], curr_info["to_node_id"]
        len2 = curr_info["segment_length"]

        # Case 1: Same arc – vehicle is traversing or dwelling on the current arc
        if prev_arc_id == curr_arc_id:
            d_net = min(d_gps, prev_info["segment_length"])
            diff = abs(d_net - d_gps)
            log_prob = -(diff / self.beta) - self.log_beta
            return float(log_prob), [curr_arc_id]

        # Case 2: Directly adjacent arcs (v1 == u2)
        if v1 == u2:
            d_net = len2
            diff = abs(d_net - d_gps)
            log_prob = -(diff / self.beta) - self.log_beta
            return float(log_prob), [curr_arc_id]

        # Case 3: Disconnected in immediate topology – search network shortest path
        path_len, path_nodes = self.get_shortest_path_info(v1, u2)

        if np.isinf(path_len):
            # Graph unreachable within cutoff
            return self.penalty_log_prob, [curr_arc_id]

        # Total network distance = path across intersections + length of target arc
        d_net = path_len + len2
        diff = abs(d_net - d_gps)
        log_prob = -(diff / self.beta) - self.log_beta

        # Reconstruct sequence of traversed arc IDs along the path nodes
        traversed_arcs = []
        for i in range(len(path_nodes) - 1):
            n_from, n_to = path_nodes[i], path_nodes[i + 1]
            edge_data = self.graph.get_edge_data(n_from, n_to)
            if edge_data:
                if isinstance(edge_data, dict):
                    first_val = next(iter(edge_data.values()))
                    if isinstance(first_val, dict) and "arc_id" in first_val:
                        traversed_arcs.append(first_val["arc_id"])
                    elif "arc_id" in edge_data:
                        traversed_arcs.append(edge_data["arc_id"])

        traversed_arcs.append(curr_arc_id)
        return float(log_prob), traversed_arcs
