# PathSnap: Final Project Executive Summary

> **Project:** PathSnap – Distributed Map-Matching & Route Reconstruction  
> **Course:** Machine Learning (Semester 5)  
> **Repository:** `https://github.com/Balreddy07/PathSnap`  
> **Date:** November 18, 2026  
> **Status:** All Phases (1 through 5) Completed, Integrated, and Verified

---

## 1. Project Goal
Transform noisy GPS observation trajectories into topologically valid, continuous road-network route sequences by combining:
1. Supervised Machine Learning (Candidate Arc Classification)
2. Bayesian Hyperparameter Optimization (Optuna GroupKFold)
3. Hidden Markov Models with Viterbi Dynamic Programming Decoding

---

## 2. Key Results at a Glance

| Milestone | Key Metric | Result |
| :--- | :--- | :---: |
| **Dataset Generation (Phase 2)** | Total Processed Records | **1,229,871 records** (208,417 GPS points across 80 trips) |
| **Dataset Splitting** | Train/Validation Split | 64 train trips (80%), 16 val trips (20%), **0 trip overlap** |
| **Phase 3 Model Benchmark** | Best Model (F1 / PR-AUC) | **Random Forest** (F1: 0.6793, AUC: 0.9353) / **XGBoost** (Recall: 0.8470, PR-AUC: 0.7584) |
| **Phase 4 Optimization** | Optimized Random Forest F1 | **0.6924** (up from 0.6793, $+1.31\%$) |
| **Phase 4 Optimization** | Optimized XGBoost F1 | **0.6725** (Evaluation Latency: 0.41s for 195k samples) |
| **Phase 5 Route Reconstruction** | Average Route Precision | **89.15%** across validation trips |
| **Phase 5 Route Reconstruction** | Average Route F1-score | **0.7187** (vs 0.6095 Geometric Baseline, $+17.9\%$ improvement) |
| **Phase 5 Route Reconstruction** | Normalized Edit Distance | **0.4212** (vs 0.5393 Geometric Baseline) |
| **Peak Single-Trip Performance** | Trip `00000005` Route F1 / Precision | **0.9073 F1** / **93.00% Precision** (Edit Distance: 0.1619) |
| **Test Suite Verification** | Automated Unit & Integration Tests | **27 passed, 0 failed in 43.67s** |

---

## 3. End-to-End Method Comparison

```
Geometric Baseline (Nearest Neighbor)      ──>  F1: 0.6095  |  Edit Distance: 0.5393  (Severe zig-zags)
ML-Only Baseline (ArgMax Prediction)       ──>  F1: 0.6752  |  Edit Distance: 0.4844  (Disconnected arcs)
PathSnap Full System (XGBoost + HMM/Viterbi) ──>  F1: 0.7187  |  Edit Distance: 0.4212  (Continuous & Topological)
```

---

## 4. Primary Artifacts Delivered

- **Core Code:**
  - `src/preprocessing/` – GPS kinematics, road network graph, candidate generation, spatial indexing.
  - `src/models/` – 5 ML model implementations (LR, SVM, RF, XGB, MLP) with feature pipeline.
  - `src/optimization/` – Optuna Bayesian GroupKFold search (`optimize_xgboost.py`, `optimize_random_forest.py`).
  - `src/hmm/` – Softmax emission probabilities, directed road shortest-path transitions, Viterbi decoder.
  - `src/visualizations/` – Interactive Folium HTML maps & 3-panel static PNG comparisons.
  - `run_pathsnap.py` – Complete end-to-end CLI.
- **Trained & Serialized Models:**
  - `models/` – Phase 3 trained models and standard scalers.
  - `outputs/phase4/models/` – Optuna-optimized `xgboost_optimized.pkl` and `random_forest_optimized.pkl`.
- **Reports:**
  - `outputs/reports/Phase1_EDA_Report.md` (Exploratory Data Analysis)
  - `outputs/phase3/reports/Phase3_Model_Training_Evaluation_Report.md` (5 ML Models)
  - `outputs/phase4/reports/Phase4_Hyperparameter_Optimization_Report.md` (Optuna Optimization)
  - `outputs/reports/PathSnap_Final_Project_Report.md` (Comprehensive Final Report)
- **Visualizations:**
  - `outputs/phase5/visualizations/00000081_interactive_route_map.html`
  - `outputs/phase5/visualizations/00000081_route_comparison.png`
  - (and trips 00000004, 00000005, 00000014, 00000018)
- **Tests:**
  - 27 unit & integration tests (`tests/`) – 100% passing.
