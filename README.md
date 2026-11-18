# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A - Machine Learning  
**Institution:** PES University, RR Campus, Department of Computer Science & Engineering  
**Project:** PathSnap – ML Hackathon  

---

## 1. Project Overview

GPS tracking systems often produce inaccurate and noisy location coordinates due to urban canyons, multipath reflections, signal attenuation, GPS dropouts, and low sampling frequencies (~1 Hz). As a result, raw trajectories appear jagged, jump across buildings, or falsely snap to parallel streets.

**PathSnap** is an end-to-end distributed machine learning system designed to reconstruct realistic vehicle trajectories by matching noisy GPS observations to physical road networks.

### System Architecture
The final system is deployed across a two-laptop distributed architecture:
* **Laptop 1 (Backend & Inference Engine):** Hosts the trained classifier, HMM/Viterbi sequential route reconstruction decoder, road-network graph representations, and a FastAPI evaluation server.
* **Laptop 2 (Evaluator Interface):** Hosts the evaluator-facing UI allowing interactive trip selection, real-time API query submission, and Leaflet/Folium geospatial visualization of reconstructed paths against ground truth.

---

## 2. Project Repository Structure

```
PathSnap/
├── data/                       # Dataset directory (excluded from Git tracking)
│   ├── raw/                    # Raw trip partitions (.track, .nodes, .arcs, .route)
│   └── processed/              # Cleaned tabular datasets and graph caches
├── notebooks/                  # Interactive exploratory and evaluation notebooks
│   └── 01_Dataset_Exploration.ipynb
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── data_loader.py          # Parsers for .track, .nodes, .arcs, .route files
│   ├── data_validation.py      # Comprehensive coordinate, topology, and trajectory checks
│   ├── dataset_statistics.py   # Dataset inventory, metrics, and aggregation
│   ├── trajectory_analysis.py  # Sampling rate, displacement, geodesic velocity & anomalies
│   ├── road_network_analysis.py# NetworkX directed graph analysis, connectivity, components
│   ├── route_analysis.py       # Ground-truth arc sequence validation & continuity checks
│   └── visualization.py        # Static distribution plots and interactive Folium maps
├── outputs/                    # Output artifacts
│   ├── csv/                    # Inventory, validation, and route statistics CSVs
│   ├── graphs/                 # High-resolution EDA distribution plots
│   ├── maps/                   # Standalone interactive Folium HTML map visualisations
│   └── reports/                # Phase technical reports and markdown summaries
├── requirements.txt            # Python environment dependencies
├── .gitignore                  # Git exclusion rules (raw data, checkpoints, venvs)
└── README.md                   # Project documentation and execution instructions
```

---

## 3. Dataset Format

Each trip is stored in an independent folder containing four whitespace-separated, headerless files:
1. **`.track`**: Raw GPS observations `[longitude, latitude, timestamp_seconds]` sampled nominally at 1 Hz.
2. **`.nodes`**: Road-network intersection coordinates `[longitude, latitude]`. The 0-based row index serves as `node_id`.
3. **`.arcs`**: Directed road-network segments `[from_node_id, to_node_id]`. The 0-based row index serves as `arc_id`.
4. **`.route`**: Chronological sequence of ground-truth `arc_id` values traversed by the vehicle.

---

## 4. Setup and Installation

### Prerequisites
* Python 3.10+ (Recommended: Python 3.11 / 3.12 / 3.14)
* Git

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Balreddy07/PathSnap.git
   cd PathSnap
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Execution Instructions (Phase 1 EDA)

To run the complete data inspection, validation, trajectory analysis, graph construction, and Folium map generation pipeline:

```bash
python -m src.data_loader
```

To explore interactively via Jupyter Notebook:
```bash
jupyter notebook notebooks/01_Dataset_Exploration.ipynb
```

All generated tabular summaries, figures, maps, and technical reports are saved in the `outputs/` directory.
