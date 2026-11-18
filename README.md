# PathSnap: Distributed Map-Matching & Route Reconstruction Using Machine Learning and Hidden Markov Models

**Course:** UE24CS352A - Machine Learning  
**Institution:** PES University, RR Campus, Department of Computer Science & Engineering  
**Project:** PathSnap – Machine Learning Hackathon  
**Repository:** [https://github.com/Balreddy07/PathSnap](https://github.com/Balreddy07/PathSnap)  
**Status:** All 5 Phases Completed, Tested, and Verified

---

## 1. Project Overview

Real-world GPS logs suffer from severe multipath errors, urban canyon reflections, temporal dropouts, and non-uniform sampling rates. Traditional geometric snapping algorithms (such as closest-point projection) fail in dense street grids, frequently matching points to parallel alleys, wrong-way one-way streets, overpasses, or disconnected segments.

**PathSnap** is an end-to-end map-matching and route reconstruction system designed to reconstruct realistic, topologically continuous vehicle trajectories by matching noisy GPS observations to physical road networks.

### System Architecture
PathSnap operates as a hybrid spatial intelligence pipeline:
1. **Local Spatial-Kinematic Classification (Phases 2–4):** A trained machine learning model (XGBoost / Random Forest) acts as an emission probability estimator $P(O_t \mid S_t)$, scoring candidate road arcs using an 11-dimensional engineered feature space capturing perpendicular distance, heading divergence, road curvature, speed dynamics, and trajectory progress.
2. **Global Network Topology Sequence Decoding (Phase 5):** A Hidden Markov Model (HMM) solved via the Viterbi dynamic programming algorithm integrates directed road network shortest paths $d_{\text{net}}$ and GPS displacement $d_{\text{gps}}$ to decode the globally optimal, topologically continuous route sequence, eliminating physically impossible jumps.

```
[Raw GPS Trajectory (.track)]        [Raw Road Network (.nodes, .arcs)]
              │                                      │
              ▼                                      ▼
    GPS Preprocessing                     Road Preprocessing
 (Outlier Removal, Kinematics)        (Directed Graph, Segment Lengths)
              │                                      │
              └───────────────┬──────────────────────┘
                              ▼
                  Spatial Indexing (RTree / KDTree)
                              │
                              ▼
                Top-K Candidate Road Arcs (k=10, r=50m)
                              │
                              ▼
                  Feature Engineering (11 Features)
                              │
                              ▼
             Optimized ML Classifier (XGBoost / RF)
                 Local Emission Probabilities P(O_t | S_t)
                              │
                              ▼
              Directed Road Network Topology Model
                 Transition Probabilities P(S_t | S_{t-1})
                              │
                              ▼
                 HMM / Viterbi Sequence Decoder
                              │
                              ▼
              Continuous Reconstructed Road Route
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    Route Metrics Evaluation        Interactive Visualizations
    (F1, Precision, Edit Dist)     (Folium HTML Maps, PNG Plots)
```

---

## 2. Benchmark Results

### 2.1 Phase 3 – Machine Learning Models (Validation Set: 194,980 samples)

| Rank | Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | 0.8812 | 0.5779 | 0.8238 | **0.6793** | **0.9353** | 0.7576 | 93.50 s |
| **2** | **XGBoost** | 0.8739 | 0.5575 | **0.8470** | **0.6724** | 0.9345 | **0.7584** | **14.66 s** |
| 3 | Multi-Layer Perceptron (MLP) | 0.8975 | 0.7117 | 0.5526 | 0.6221 | 0.9104 | 0.6859 | 74.20 s |
| 4 | Support Vector Machine (LinearSVC) | **0.9017** | **0.7745** | 0.5030 | 0.6099 | 0.9234 | 0.7206 | 6.80 s |
| 5 | Logistic Regression | 0.8274 | 0.4655 | 0.8773 | 0.6082 | 0.9237 | 0.7188 | 1.84 s |

### 2.2 Phase 4 – Optuna Bayesian Optimization (GroupKFold CV)

| Model Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest (Baseline Phase 3) | 0.8812 | 0.5779 | 0.8238 | 0.6793 | **0.9353** | 0.7576 |
| **Random Forest (Optimized Phase 4)** | **0.8947** | **0.6251** | 0.7759 | **0.6924** | 0.9332 | 0.7526 |
| XGBoost (Baseline Phase 3) | 0.8739 | 0.5575 | **0.8470** | 0.6724 | 0.9345 | **0.7584** |
| **XGBoost (Optimized Phase 4)** | 0.8770 | 0.5667 | 0.8269 | 0.6725 | 0.9310 | 0.7505 |

### 2.3 Phase 5 – Route Reconstruction Benchmark (Validation Trips)

| Method | Route Precision | Route Recall | Route F1-Score | Normalized Edit Distance |
| :--- | :---: | :---: | :---: | :---: |
| **Geometric Nearest Neighbor Baseline** | 70.05% | 61.69% | 0.6095 | 0.5393 (High zig-zagging) |
| **ML-Only Snapping Baseline** | 78.15% | 66.74% | 0.6752 | 0.4844 (Disconnected arcs) |
| **PathSnap Full System (XGBoost + HMM)** | **89.15%** | **64.40%** | **0.7187** | **0.4212** (Continuous route) |

*Single-Trip Peak:* Trip `00000005` achieved **0.9073 Route F1**, **93.00% Precision**, and **0.1619 Normalized Edit Distance**.

---

## 3. Repository Structure

```
PathSnap/
├── configs/
│   ├── preprocessing.yaml            # Phase 2 spatial indexing and feature configurations
│   ├── phase3_models.yaml            # Phase 3 model hyperparameters and training parameters
│   └── phase4_config.yaml            # Phase 4 Optuna GroupKFold search space
├── data/
│   ├── raw/train/                    # 80 authentic raw trip folders (.track, .nodes, .arcs, .route)
│   └── processed/
│       ├── train.csv                 # 1,034,891 training records across 64 trips
│       └── val.csv                   # 194,980 validation records across 16 trips
├── models/                           # Phase 3 serialized models and standard scalers
│   ├── logistic_regression/
│   ├── svm/
│   ├── random_forest/
│   ├── xgboost/
│   └── mlp/
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   └── 03_Phase3_Model_Training_Evaluation.ipynb
├── outputs/
│   ├── phase3/                       # Phase 3 metrics, predictions, plots, reports
│   ├── phase4/                       # Phase 4 Optuna histories, best params, optimized models
│   │   ├── models/xgboost_optimized.pkl
│   │   ├── models/random_forest_optimized.pkl
│   │   └── reports/Phase4_Hyperparameter_Optimization_Report.md
│   ├── phase5/                       # Phase 5 route reconstructions and visualizations
│   │   ├── reconstructed_routes/     # Reconstructed route JSON files
│   │   ├── visualizations/           # Interactive Folium HTML maps & PNG plots
│   │   └── route_metrics/            # Comparative route evaluation CSVs
│   └── reports/                      # Comprehensive technical reports
│       ├── Phase1_EDA_Report.md
│       ├── PathSnap_Final_Project_Report.md
│       └── PathSnap_Final_Project_Summary.md
├── src/
│   ├── preprocessing/                # GPS kinematics, spatial index, candidates, road preprocessing
│   ├── models/                       # 5 ML model implementations and trainers
│   ├── optimization/                 # Optuna GroupKFold tuning (XGBoost & Random Forest)
│   ├── hmm/                          # Emission probabilities, road network transitions, Viterbi
│   ├── evaluation/                   # Route precision, recall, F1, Levenshtein edit distance
│   └── visualizations/               # Multi-layer Folium maps and static 3-panel plots
├── tests/                            # Comprehensive automated test suite (27 tests)
├── run_pathsnap.py                   # Unified end-to-end command-line interface
├── requirements.txt
└── README.md
```

---

## 4. Setup and Installation

### Prerequisites
* Python 3.10+ (Tested on Python 3.14)
* Virtual environment tool (`venv`)

### Installation Steps
```bash
git clone https://github.com/Balreddy07/PathSnap.git
cd PathSnap

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 5. Execution Instructions

### Run End-to-End Route Reconstruction (CLI)
```bash
# Reconstruct and visualize a specific validation trip (e.g. 00000081):
python run_pathsnap.py --trip-id 00000081 --visualize

# Evaluate across multiple validation trips:
python run_pathsnap.py --evaluate --limit 5 --visualize

# Use Random Forest instead of XGBoost:
python run_pathsnap.py --trip-id 00000005 --model random_forest --visualize
```

### Run Model Training & Hyperparameter Optimization
```bash
# Train and evaluate all 5 Phase 3 models:
python src/models/train_all.py
python src/models/evaluate_models.py
python src/models/compare_models.py

# Run Phase 4 Optuna Bayesian optimization:
python src/optimization/run_phase4.py
```

### Run Distributed Two-Laptop Evaluation
```bash
# On Laptop 1 (Inference Server):
python src/api/server.py --host 0.0.0.0 --port 8000

# On Laptop 2 (Evaluator):
# Open browser at: http://<LAPTOP_1_IP>:8000
# Or run terminal client:
python src/api/client.py --server http://<LAPTOP_1_IP>:8000 --trip-id 00000081
```
*Complete Step-by-Step Guide:* [docs/TWO_LAPTOP_DISTRIBUTED_SETUP_GUIDE.md](docs/TWO_LAPTOP_DISTRIBUTED_SETUP_GUIDE.md)

### Run Test Suite
```bash
pytest tests/ -v
# Output: 27 passed in ~43 seconds
```

---

## 6. Visualizations

Generated visualizations are stored under `outputs/phase5/visualizations/`:
- **Interactive Multi-Layer Folium Map (`.html`):** Overlays raw GPS trajectory fixes, ground-truth route sequence, and reconstructed PathSnap route with toggleable layers.
- **3-Panel Static Comparison Plot (`.png`):**
  1. Panel 1: Raw GPS point scatter and trajectory.
  2. Panel 2: Ground-truth road network sequence.
  3. Panel 3: PathSnap Viterbi reconstruction overlaid on ground-truth with metric callouts.

---

## 7. Automated Test Suite Summary

| Test Module | Coverage | Status |
| :--- | :--- | :---: |
| `tests/test_phase3_data_loading.py` | Config loading, CSV existence, 0 trip overlap, target derivation, positive rate | **PASSED (6/6)** |
| `tests/test_phase3_features.py` | Feature presence, training-only scaling, no NaN/inf values | **PASSED (3/3)** |
| `tests/test_phase3_target.py` | Target binary values, matching logic, non-matching logic | **PASSED (3/3)** |
| `tests/test_phase3_no_leakage.py` | Strict leakage check on ground-truth and target columns | **PASSED (2/2)** |
| `tests/test_phase3_evaluation.py` | Metrics values range, predictions consistency | **PASSED (4/4)** |
| `tests/test_phase4_optimization.py` | Hyperparameter file existence, model loading, shape validation, comparison table | **PASSED (4/4)** |
| `tests/test_phase5_hmm.py` | Emission normalization, transition calculation, Viterbi lattice decoding, edit distance | **PASSED (4/4)** |
| `tests/test_end_to_end.py` | Full pipeline integration on authentic trip | **PASSED (1/1)** |
| **TOTAL** | **27 Automated Tests** | **100% PASSED** |
