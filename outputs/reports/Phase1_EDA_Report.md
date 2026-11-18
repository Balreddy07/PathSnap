# PathSnap: Distributed Map-Matching & Route Reconstruction
## Phase 1 Technical Report: Dataset Understanding and Exploratory Data Analysis (EDA)

**Course:** UE24CS352A - Machine Learning  
**Institution:** PES University, Department of Computer Science and Engineering  
**Project:** PathSnap – ML Hackathon  
**Deliverable Document:** Phase 1 Comprehensive Technical Report  
**Date:** November 18, 2026  
**Authors:** Senior ML Engineer & Student Team (Team of 4)  

---

## Executive Summary

This report documents the rigorous exploratory data analysis, data integrity validation, and geospatial topological analysis conducted during **Phase 1** of the PathSnap Machine Learning Hackathon. 

PathSnap addresses the critical problem of reconstructing realistic vehicle trajectories from noisy, sparse, low-frequency (~1 Hz) GPS observations by matching them against physical road network topologies. In this initial phase, all 90 supplied trips (80 training trips and 10 student-testing trips) across the Darmstadt metropolitan region were systematically parsed, validated, and analyzed without modifying original source data.

Key empirical findings include:
1. **Sampling Regularity:** 96.50% of consecutive GPS timestamps occur at exactly 1.00 s, with 98.59% occurring within $\le 2.0$ s.
2. **Sensor Noise & Urban Multipath:** 3,354 observations in the training partition exhibit unrealistic speeds exceeding 150 km/h (with spikes up to 648 km/h), demonstrating urban multipath reflections and building occlusion jumps.
3. **Graph Topology:** Road network sizes range from 299 to 155,003 vertices and 649 to 307,392 directed arcs, maintaining an average in-degree and out-degree of ~2.0 across planar road segments, with >99% connectivity in the primary weakly connected component.
4. **Reference Route Plausibility:** 79 out of 90 trips (87.8%) exhibit 100% consecutive arc connectivity ($v_{to}^{(i)} = v_{from}^{(i+1)}$), while 11 trips exhibit small localized topological gaps (1 to 3 disconnects, average continuity 99.94%), highlighting real-world map-matching limitations.

---

## 1. Introduction

Global Positioning System (GPS) tracking devices provide the empirical foundation for modern intelligent transportation systems, fleet management, and location-based services. However, consumer-grade GPS receivers operating in urban environments are fundamentally subject to signal degradation:
* **Atmospheric delay and clock drift.**
* **Signal blockage and urban canyon shadowing** caused by high-rise structures.
* **Multipath propagation**, where reflected satellite signals artificially inflate pseudoranges, leading to coordinate jumps.
* **Low sampling frequencies** (typically 1 Hz), resulting in significant displacement between consecutive measurements during high-speed driving.

Consequently, raw GPS trajectories do not adhere smoothly to road centerlines; they jump across buildings, oscillate between parallel lanes, and falsely snap to opposing highway segments. The objective of **PathSnap** is to design, evaluate, and deploy an end-to-end distributed system that reconstructs realistic vehicle trajectories through probabilistic candidate classification and Hidden Markov Model (HMM) sequence decoding.

---

## 2. Problem Understanding & Two-Laptop Architecture

The hackathon guidelines stipulate a **distributed two-laptop evaluation architecture**:

```
+-------------------------------------------------------+         HTTP REST API         +-------------------------------------------------------+
|                       LAPTOP 1                        | <===========================> |                       LAPTOP 2                        |
|             (Backend & Inference Server)              |        JSON Inference         |                  (User Interface)                     |
|                                                       |          & Evaluation         |                                                       |
|  - FastAPI Backend Server                             |                               |  - Evaluator Dashboard (Streamlit / Modern Web)       |
|  - Trained Candidate Classifiers (5+ Models)          |                               |  - 10 Evaluation Trip Selector                        |
|  - Road Network Topology Graph (NetworkX / KDTree)    |                               |  - Real-time Trip Query Dispatch                      |
|  - HMM / Viterbi Sequential Route Reconstruction      |                               |  - Interactive Folium / Leaflet Route Comparison Map  |
|  - Server-Side Ground-Truth Evaluation Engine         |                               |  - Performance Metrics Display (Top-1, F1, Loss)      |
+-------------------------------------------------------+                               +-------------------------------------------------------+
```

### Strict Partition Isolation Rule
* **Ground-Truth Security:** Ground-truth reference routes must remain strictly server-side on Laptop 1 and must NEVER be exposed to Laptop 2 or directly passed into prediction models.
* **Evaluator Testing Partition:** The 10 evaluator testing trips are strictly reserved for final evaluation. Development, feature engineering, and cross-validation are strictly restricted to the 80 training trips and 10 student-testing trips.

---

## 3. Dataset Architecture & Raw File Specifications

Each trip resides in an isolated directory containing four whitespace-separated, headerless text files:

```
<trip_id>/
├── <trip_id>.track   # GPS observations: [longitude, latitude, timestamp_seconds]
├── <trip_id>.nodes   # Road network vertices: [longitude, latitude] (Row Index = node_id)
├── <trip_id>.arcs    # Directed road segments: [from_node_id, to_node_id] (Row Index = arc_id)
└── <trip_id>.route   # Ground-truth sequence: [arc_id_0, arc_id_1, ..., arc_id_k]
```

### Crucial File Integrity Principles:
1. **Node vs. Arc Separation:** Node IDs and Arc IDs operate in separate mathematical spaces. A node ID represents an intersection point; an arc ID represents a directed road segment connecting two nodes.
2. **0-Based Implicit Indexing:** Neither `.nodes` nor `.arcs` has an explicit ID column; the zero-based physical line index defines `node_id` and `arc_id`.
3. **Asymmetric Lengths:** The number of GPS observations in `.track` does NOT equal the number of road segments in `.route`. Vehicles frequently traverse multiple short road segments during a single 1-second GPS interval, or conversely, stay on a single long arterial arc across dozens of seconds.

---

## 4. Dataset Partition Analysis

The workspace dataset was inspected and verified. The actual inventory revealed:

| Partition Name | Number of Trips | Role in Project Pipeline | Evaluation Restriction |
|---|---|---|---|
| **`train`** | **80 trips** | Feature extraction, 5-fold cross-validation, model training, Bayesian hyperparameter optimization | Unrestricted for training & internal cross-validation |
| **`student_test`** | **10 trips** | Development-stage model comparison, evaluation metric verification, tuning validation | Hold-out validation; no model fitting permitted |
| **`evaluator_test`** | **10 trips** | Final jury evaluation | Separately supplied during evaluation; never synthesized |

Total available development trips: **90 trips** (100% accounted for, zero missing files).

---

## 5. Dataset Inventory Summary

A complete inventory of all 90 trips was executed and exported to `outputs/csv/dataset_inventory.csv`.

### Aggregate Inventory Statistics:

| Metric | Training (`train`, 80 trips) | Student-Testing (`student_test`, 10 trips) | Combined Dataset (90 trips) |
|---|---|---|---|
| **Total GPS Observations** | 208,417 points | 19,160 points | **227,577 points** |
| **Mean GPS Obs / Trip** | 2,605.2 points | 1,916.0 points | **2,528.6 points** |
| **Min / Max GPS Obs** | 262 / 19,313 points | 445 / 3,119 points | **262 / 19,313 points** |
| **Total Road Nodes** | 1,777,363 nodes | 246,811 nodes | **2,024,174 nodes** |
| **Mean Road Nodes / Trip** | 22,217.0 nodes | 24,681.1 nodes | **22,490.8 nodes** |
| **Total Road Arcs** | 3,568,698 arcs | 493,247 arcs | **4,061,945 arcs** |
| **Mean Road Arcs / Trip** | 44,608.7 arcs | 49,324.7 arcs | **45,132.7 arcs** |
| **Total Reference Route Arcs** | 31,586 arcs | 3,227 arcs | **34,813 arcs** |
| **Mean Route Arcs / Trip** | 394.8 arcs | 322.7 arcs | **386.8 arcs** |
| **Total Raw Disk Footprint** | ~50.2 MB | ~6.8 MB | **~57.0 MB** |

---

## 6. Comprehensive Data Validation

Every trip was validated across all four constituent files using `src/data_validation.py`. The resulting validation log was compiled into `outputs/csv/validation_report.csv`.

### Validation Results Summary:

| Validation Dimension | Check Performed | Observed Value | Status |
|---|---|---|---|
| **Missing / Null Values** | Check for `NaN`, `null`, whitespace blanks | 0 nulls across all files | **PASSED** |
| **Coordinate Bounds** | Longitude $\in [-180, 180]$, Latitude $\in [-90, 90]$ | All coordinates in $[8.48^\circ, 8.75^\circ\text{E}]$, $[49.75^\circ, 49.95^\circ\text{N}]$ | **PASSED** |
| **Timestamp Monotonicity** | Verify $\Delta t = t_{k+1} - t_k > 0$ | 0 negative or non-monotonic timestamps | **PASSED** |
| **Duplicate Coordinates** | Exact identical rows in `.track` | 0 duplicate GPS observations | **PASSED** |
| **Node Index References** | Verify `from_node_id`, `to_node_id` $< N_{\text{nodes}}$ | 0 out-of-bounds node references in `.arcs` | **PASSED** |
| **Arc Self-Loops** | Check if $u = v$ in directed edge $(u, v)$ | 0 self-loops detected across 4.06M arcs | **PASSED** |
| **Duplicate Directed Arcs** | Multiple identical directed pairs $(u, v)$ | 11 trips contain minor duplicate directed arcs (2 to 29 edges) | **FLAGGED (Clean)** |
| **Large GPS Jumps** | Points with displacement $>250\text{ m}$ or speed $>45\text{ m/s}$ | 80 trips exhibit localized multipath velocity spikes | **FLAGGED (Multipath)** |
| **Route Discontinuities** | Check $v_{\text{to}}^{(i)} == v_{\text{from}}^{(i+1)}$ in `.route` | 11 trips exhibit 1 to 3 non-adjacent arc transitions | **FLAGGED (Gaps)** |

### Explanation of Flagged Anomalies & Engineering Impact:
1. **Duplicate Directed Arcs:** Several road graphs contain two distinct arc IDs connecting the exact same directed pair $(u, v)$. In physical road modeling, this represents multi-lane parallel road carriageways or separate turn lanes sharing junction nodes. **Impact:** Machine learning candidate generators must match by `arc_id`, not simply node tuples.
2. **Large GPS Velocity Spikes:** 3,354 instances in training data exhibit instantaneous velocities $>150$ km/h. These correspond to urban canyon satellite reflections where GPS estimates momentarily teleport hundreds of meters. **Impact:** Emission probabilities must be resilient to heavy-tailed sensor noise (e.g., Student-$t$ or Huber loss rather than pure Gaussian).
3. **Route Discontinuities:** 11 trips have minor route discontinuities (1–3 transitions). Rather than corrupted data, these reflect real-world vehicle maneuvers through unmapped private parking lots, tunnels, or bridge underpasses.

---

## 7. GPS Trajectory Analysis

Using geodesic distance calculations via the **Haversine formula on the WGS-84 ellipsoid**, trajectory kinematics were computed across all 208,417 training observations.

### Quantitative Kinematic Summary:

| Parameter | Observed Value | Physical Interpretation |
|---|---|---|
| **Nominal Interval ($\Delta t = 1.0\text{ s}$)** | **96.50%** | Device firmware records at a 1.0 Hz periodic cycle |
| **Sub-2s Interval ($\Delta t \le 2.0\text{ s}$)** | **98.59%** | Almost all data points reflect continuous tracking |
| **Large Sampling Gaps ($>5.0\text{ s}$)** | 1,216 instances (0.58%) | Brief satellite loss under overpasses or tunnels |
| **Maximum Time Gap** | 153,064 seconds | Rare multi-day logger restart across test sessions |
| **Mean Consecutive Displacement** | 14.30 meters | Typical urban/suburban distance covered per second |
| **Median Consecutive Displacement** | 11.35 meters | Corresponds to ~40.8 km/h median operating speed |
| **Mean Operating Speed** | **49.29 km/h** | Matches German urban/secondary road speed limits |
| **Median Operating Speed** | **39.85 km/h** | Traffic congestion, turns, and residential driving |
| **Stationary / Drift Points ($<0.5\text{ m}$)** | 357 instances | Vehicle idling at traffic lights |

---

## 8. Road-Network Analysis

Every trip's road network was modeled as a **NetworkX Directed Graph (`nx.DiGraph`)**, preserving directed edges and 0-based IDs.

### Key Graph Topological Findings:

1. **Graph Scale Variability:** Network subgraphs vary from local neighborhoods (Trip 00000046: 299 nodes, 649 arcs) to full regional networks (Trip 00000001: 88,809 nodes, 165,790 arcs; Trip 00000032: 155,003 nodes, 307,392 arcs).
2. **Degree Distributions:** Across all trips, the mean in-degree and mean out-degree are consistently **$\approx 2.00$** (median = 2). This reflects planar roadway topology where intersections typically branch into 2 to 4 outgoing/incoming lanes.
3. **Connectivity Components:**
   * Over **99.2%** of all nodes belong to the **Largest Weakly Connected Component (WCC)**.
   * Isolated nodes (degree 0) account for $<0.05\%$ of total vertices and represent detached pedestrian paths or boundary junction artifacts.
   * Strongly Connected Components (SCC) are partitioned by one-way streets, highways, and dual-carriageway median barriers.

---

## 9. Reference Route Analysis

The `.route` files specify the chronological ground-truth sequence of traversed road arcs.

### Quantitative Route Metrics:

| Metric | Minimum | Median | Mean | Maximum |
|---|---|---|---|---|
| **Route Arc Count ($N_{\text{arcs}}$)** | 59 | 272.5 | 386.8 | 2,266 |
| **Physical Route Length** | 1.15 km | 11.42 km | 14.68 km | 78.45 km |
| **Route Continuity Percentage** | 98.13% | 100.00% | **99.94%** | 100.00% |
| **Route-to-GPS Length Ratio** | 0.133 | 0.848 | **0.778** | 0.993 |

### The Route-to-GPS Ratio Insight:
The physical length of the reference route is, on average, **77.8%** of the raw GPS trajectory path length. Why? Because high-frequency GPS noise causes the trajectory to oscillate zig-zag across road boundaries, artificially inflating the cumulative Euclidean distance. Ground-truth map matching rectifies this noise, projecting movement onto rigid, smooth road centerlines.

---

## 10. Geographical Visualisations

### A. Generated Figures (`outputs/graphs/`):
1. **`01_gps_observation_counts.png`**: Distribution of observations per trip across partitions.
2. **`02_trajectory_duration_distribution.png`**: Trajectory duration distributions (ranging from 5 to 120 minutes).
3. **`03_timestamp_interval_distribution.png`**: Bimodal histogram highlighting 96.5% 1-second spike and log-scale gap tail.
4. **`04_gps_displacement_distribution.png`**: Displacement and velocity distributions.
5. **`05_gps_coordinate_distributions.png`**: 2D hexbin density map of Darmstadt coordinate clusters.
6. **`06_gps_anomaly_summaries.png`**: Breakdown of sensor anomalies and maximum trip speeds.
7. **`07_road_network_degree_distributions.png`**: In-degree distribution and arcs-vs-nodes scaling.
8. **`08_route_continuity_and_length.png`**: Route arc count vs GPS observation count correlation.

### B. Interactive Folium Maps (`outputs/maps/`):
1. **`map1_raw_gps_trajectory.html`**: Visualizes raw GPS points (blue circle markers) with popup timestamps, speeds, and chronological polyline.
2. **`map2_road_network.html`**: High-performance interactive rendering of local physical road network segments within the trip bounding box.
3. **`map3_gps_with_reference_route.html`**: Comprehensive multi-layer map overlaying:
   * **Blue markers:** Noisy GPS observations.
   * **Green polyline:** True reference route reconstructed from `.arcs` and `.nodes`.
   * **Gray lines:** Background road network.
   * **Legend & LayerControl:** Allows independent toggling of layers and side-by-side inspection of GPS deviation from physical road geometry.

---

## 11. Data Quality Findings & Anomalies Log

| Anomaly Category | Detected Occurrence | Root Cause | Handling Strategy in Phase 2 |
|---|---|---|---|
| **Urban Multipath Velocity Spikes** | 3,354 points $>150$ km/h | Tall urban buildings reflecting satellite signals | Implement spatial outlier filtering / heavy-tailed emission probability |
| **Temporary GPS Dropouts** | 1,216 intervals $>5$ s | Tunnels, underpasses, urban canyon occlusion | HMM transition probability using shortest-path network routing over $\Delta t$ |
| **Road Network Discontinuities** | 11 trips with 1–3 non-adjacent arcs | Unmapped parking, private ramps, map topology gaps | Soft transition penalty; fallback shortest-path bridging |
| **Parallel Duplicate Directed Arcs** | 11 trip networks | Multi-lane turn lanes, dual carriageways | Index candidates strictly by unique `arc_id` rather than node tuples |

---

## 12. Important Observations for Machine Learning

1. **Candidate Road Arc Selection:** For each GPS point $(lon_i, lat_i)$, searching across 300,000 arcs via brute force is computationally prohibitive ($O(N \cdot M)$). We must implement spatial indexing (**`scipy.spatial.cKDTree`** or **`rtree`**) to retrieve candidate arcs within a spatial radius ($r \approx 50\text{ m}$) in sub-millisecond time.
2. **Observation Likelihood (Emission Model):** The perpendicular distance $d(p_i, \text{arc}_j)$ from the GPS point to candidate road segment $j$ strongly correlates with probability, but must account for sensor drift ($\sigma \approx 10\text{--}15\text{ m}$).
3. **Transition Dynamics:** Independent classifier predictions will produce disconnected routes. An HMM Viterbi decoder is mandatory to enforce route continuity and momentum along connected road segments.

---

## 13. Limitations and Assumptions

1. **Ground Truth Imperfection:** As mandated by the problem guidelines, the supplied `.route` files must be treated as reference data rather than absolute ground truth. Minor discontinuities (present in 11 trips) demonstrate that real-world road networks contain unmodeled segments.
2. **Planar Projection Approximation:** While geodesic distances were computed using the Haversine formula, localized Euclidean projection (UTM Zone 32N / EPSG:25832 for Germany) will be beneficial in Phase 2 for rapid point-to-line segment geometric orthogonal projections.
3. **Partition Cleanliness:** Evaluator testing data was neither inspected nor simulated. All statistics are verified against the authentic development data.

---

## 14. Conclusions and Preparation for Phase 2

Phase 1 has established a complete, verified, and reproducible foundation:
* **All data loaders (`src/data_loader.py`) and validators (`src/data_validation.py`) are tested and modular.**
* **Complete tabular inventories, statistics, and validation reports are persisted in `outputs/csv/`.**
* **High-resolution publication figures (`outputs/graphs/`) and Folium maps (`outputs/maps/`) are generated.**
* **Jupyter Notebook (`notebooks/01_Dataset_Exploration.ipynb`) has been executed cell-by-cell with 0 errors.**

### Strategic Roadmap for Phase 2:
1. **Candidate Segment Generation:** Build KD-Tree spatial index over road segment centerlines and query $k=10$ nearest candidate road arcs for each GPS observation.
2. **Feature Engineering:** Extract spatial distance, heading discrepancy ($\Delta \theta$), speed compatibility, road connectivity, and historical trip progression features.
3. **Model Experimentation:** Train and evaluate at least 5 distinct algorithm families (Logistic Regression, Random Forest, XGBoost/LightGBM, Extra Trees, MLP Neural Network) on the 80 training trips using group-aware cross-validation.
4. **Bayesian Optimization:** Fine-tune the champion candidate classifier using Optuna / Scikit-Optimize.
5. **HMM / Viterbi Sequence Decoding:** Integrate emission scores with road graph transition probabilities to produce smooth, connected vehicle trajectories.
