"""
src/api/server.py
=================
Distributed Map-Matching & Route Reconstruction FastAPI Server.
Runs on Laptop 1 (Inference Server) and serves predictions, metrics,
and interactive Leaflet/Folium visualization dashboards to Laptop 2 (Evaluator).
"""

import os
import sys
import json
import socket
import argparse
from typing import Optional, List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".mplconfig"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.hmm.route_reconstruction import reconstruct_trip_route
from src.visualizations.route_map_generator import (
    generate_interactive_route_map,
    generate_static_route_comparison_plot,
)

app = FastAPI(
    title="PathSnap Distributed Map-Matching Inference Server",
    description="Backend inference and evaluation engine for distributed GPS route reconstruction.",
    version="1.0.0",
)

# Enable CORS for cross-laptop requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "train")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "phase5")
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "phase4", "models", "xgboost_optimized.pkl")

# Cached model instance
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import pickle
        if not os.path.exists(MODEL_PATH):
            fallback = os.path.join(PROJECT_ROOT, "models", "xgboost", "model.pkl")
            if os.path.exists(fallback):
                with open(fallback, "rb") as f:
                    _MODEL = pickle.load(f)
            else:
                raise FileNotFoundError("Model file not found. Run model training first.")
        else:
            with open(MODEL_PATH, "rb") as f:
                _MODEL = pickle.load(f)
    return _MODEL


def get_local_ip() -> str:
    """Detects local LAN IP address for easy multi-laptop connection."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status."""
    return {
        "status": "HEALTHY",
        "service": "PathSnap Map-Matching Inference Server",
        "model_loaded": os.path.exists(MODEL_PATH),
        "local_ip": get_local_ip(),
        "port": 8000,
    }


@app.get("/trips")
def list_available_trips():
    """Returns list of available trips from dataset."""
    if not os.path.exists(RAW_ROOT):
        return {"trips": []}
    trips = sorted([d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))])
    return {
        "total_trips": len(trips),
        "sample_trips": trips[:20],
        "validation_recommended": ["00000081", "00000005", "00000004", "00000014", "00000018"],
    }


@app.post("/reconstruct/{trip_id}")
def reconstruct_route(
    trip_id: str,
    beta: float = Query(20.0, description="HMM transition beta parameter (meters)"),
    search_radius_m: float = Query(50.0, description="Candidate search radius in meters"),
    max_k: int = Query(10, description="Max candidate arcs per GPS fix"),
):
    """
    Executes end-to-end map-matching on a specific trip.
    Returns reconstructed arcs, baseline comparison metrics, and status.
    """
    try:
        model = get_model()
        formatted_tid = f"{int(trip_id):08d}" if str(trip_id).isdigit() else str(trip_id)
        res = reconstruct_trip_route(
            trip_id=formatted_tid,
            raw_root=RAW_ROOT,
            model=model,
            beta=beta,
            search_radius_m=search_radius_m,
            max_k=max_k,
        )

        if res.get("status") == "NO_CANDIDATES":
            raise HTTPException(status_code=404, detail=f"No candidates found for trip {trip_id}")

        # Ensure visualizations are generated
        vis_dir = os.path.join(OUT_DIR, "visualizations")
        html_path = os.path.join(vis_dir, f"{formatted_tid}_interactive_route_map.html")
        png_path  = os.path.join(vis_dir, f"{formatted_tid}_route_comparison.png")
        generate_interactive_route_map(res, html_path)
        generate_static_route_comparison_plot(res, png_path)

        return {
            "trip_id": formatted_tid,
            "duration_s": res["duration_s"],
            "n_observations": res["n_observations"],
            "n_candidates_total": res["n_candidates_total"],
            "n_fallbacks": res["n_fallbacks"],
            "viterbi_score": res["viterbi_score"],
            "metrics_hmm": res["metrics_hmm"],
            "metrics_ml_baseline": res["metrics_ml_baseline"],
            "metrics_geometric_baseline": res["metrics_geometric_baseline"],
            "reconstructed_arcs_count": len(res["reconstructed_route"]),
            "ground_truth_arcs_count": len(res["ground_truth_route"]),
            "map_url": f"/visualize/{formatted_tid}",
            "plot_url": f"/plot/{formatted_tid}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualize/{trip_id}", response_class=HTMLResponse)
def get_trip_map(trip_id: str):
    """Serves the interactive Folium HTML map for a trip."""
    formatted_tid = f"{int(trip_id):08d}" if str(trip_id).isdigit() else str(trip_id)
    html_path = os.path.join(OUT_DIR, "visualizations", f"{formatted_tid}_interactive_route_map.html")
    if not os.path.exists(html_path):
        # Auto-reconstruct on the fly if not cached
        reconstruct_route(formatted_tid)

    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="Interactive map not found.")


@app.get("/plot/{trip_id}")
def get_trip_plot(trip_id: str):
    """Serves the 3-panel static comparison plot PNG."""
    formatted_tid = f"{int(trip_id):08d}" if str(trip_id).isdigit() else str(trip_id)
    png_path = os.path.join(OUT_DIR, "visualizations", f"{formatted_tid}_route_comparison.png")
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found.")


@app.get("/metrics/summary")
def get_metrics_summary():
    """Returns comparative summary CSV data across all evaluated trips."""
    csv_path = os.path.join(OUT_DIR, "route_metrics", "route_evaluation_comparison.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    return {"message": "No batch evaluation summary yet. Run evaluation first."}


@app.get("/", response_class=HTMLResponse)
def evaluator_dashboard():
    """
    Evaluator Web Dashboard served to Laptop 2 browser.
    Allows live interactive evaluation, trip selection, metrics display,
    and embedded map rendering.
    """
    local_ip = get_local_ip()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>PathSnap – Evaluator Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ background-color: #0f172a; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    .navbar {{ background-color: #1e293b; border-bottom: 1px solid #334155; }}
    .card {{ background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 20px; }}
    .metric-val {{ font-size: 2rem; font-weight: 700; color: #38bdf8; }}
    .metric-label {{ font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    .btn-primary {{ background-color: #0284c7; border: none; font-weight: 600; padding: 10px 24px; border-radius: 8px; }}
    .btn-primary:hover {{ background-color: #0369a1; }}
    iframe {{ width: 100%; height: 600px; border: 1px solid #334155; border-radius: 12px; }}
    .badge-hmm {{ background-color: #059669; }}
    .badge-geo {{ background-color: #d97706; }}
    .badge-ml {{ background-color: #7c3aed; }}
    pre {{ background: #0f172a; padding: 12px; border-radius: 8px; color: #38bdf8; }}
  </style>
</head>
<body>
  <nav class="navbar navbar-dark px-4 py-3">
    <div class="container-fluid">
      <span class="navbar-brand mb-0 h1 fw-bold">🚀 PathSnap – Distributed Map-Matching Engine</span>
      <span class="badge bg-secondary">Inference Server: {local_ip}:8000</span>
    </div>
  </nav>

  <div class="container-fluid px-4 py-4">
    <div class="row">
      <!-- Control Panel -->
      <div class="col-lg-4">
        <div class="card p-4">
          <h4 class="card-title fw-bold text-white mb-3">Trip Evaluation Controller</h4>
          <p class="text-secondary small">Connected to Laptop 1 backend. Select a trip ID to reconstruct and benchmark.</p>
          
          <div class="mb-3">
            <label class="form-label text-light fw-semibold">Select or Enter Trip ID</label>
            <select id="tripSelect" class="form-select bg-dark text-light border-secondary mb-2" onchange="onSelectTrip()">
              <option value="00000081">00000081 (Validation – Long Route, 3224 pts)</option>
              <option value="00000005">00000005 (Validation – F1: 0.907, 1045 pts)</option>
              <option value="00000004">00000004 (Validation – Urban, 1064 pts)</option>
              <option value="00000014">00000014 (Validation – 448 pts)</option>
              <option value="00000018">00000018 (Validation – 379 pts)</option>
            </select>
            <input type="text" id="customTrip" class="form-control bg-dark text-light border-secondary" placeholder="Or custom trip ID (e.g. 00000001)">
          </div>

          <div class="row g-2 mb-3">
            <div class="col-6">
              <label class="form-label small text-secondary">HMM Beta (m)</label>
              <input type="number" id="paramBeta" class="form-control bg-dark text-light border-secondary" value="20.0" step="5">
            </div>
            <div class="col-6">
              <label class="form-label small text-secondary">Radius (m)</label>
              <input type="number" id="paramRadius" class="form-control bg-dark text-light border-secondary" value="50.0" step="10">
            </div>
          </div>

          <button id="runBtn" class="btn btn-primary w-100" onclick="triggerReconstruction()">
            ⚡ Run Map-Matching & Reconstruct
          </button>

          <div id="loading" class="text-center mt-3 d-none">
            <div class="spinner-border text-info" role="status"></div>
            <p class="small text-info mt-2">Computing candidate emissions & running Viterbi decoding...</p>
          </div>
        </div>

        <!-- Metrics Cards -->
        <div id="metricsPanel" class="card p-4 d-none">
          <h5 class="fw-bold text-white mb-3">Reconstruction Metrics</h5>
          
          <div class="row g-2 text-center mb-3">
            <div class="col-4">
              <div class="p-2 border border-secondary rounded">
                <div class="metric-val text-success" id="valF1">-</div>
                <div class="metric-label">HMM F1</div>
              </div>
            </div>
            <div class="col-4">
              <div class="p-2 border border-secondary rounded">
                <div class="metric-val text-info" id="valPrecision">-</div>
                <div class="metric-label">Precision</div>
              </div>
            </div>
            <div class="col-4">
              <div class="p-2 border border-secondary rounded">
                <div class="metric-val text-warning" id="valEditDist">-</div>
                <div class="metric-label">Edit Dist</div>
              </div>
            </div>
          </div>

          <table class="table table-dark table-sm table-bordered text-center small mb-0">
            <thead>
              <tr>
                <th>Method</th>
                <th>F1-Score</th>
                <th>Edit Dist</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span class="badge badge-hmm">PathSnap HMM</span></td>
                <td id="tableHmmF1" class="fw-bold text-success">-</td>
                <td id="tableHmmDist">-</td>
              </tr>
              <tr>
                <td><span class="badge badge-ml">ML-Only</span></td>
                <td id="tableMlF1">-</td>
                <td id="tableMlDist">-</td>
              </tr>
              <tr>
                <td><span class="badge badge-geo">Geometric</span></td>
                <td id="tableGeoF1">-</td>
                <td id="tableGeoDist">-</td>
              </tr>
            </tbody>
          </table>
          <p class="small text-secondary mt-2 mb-0" id="timeStats"></p>
        </div>
      </div>

      <!-- Map & Visualization Panel -->
      <div class="col-lg-8">
        <div class="card p-4">
          <div class="d-flex justify-content-between align-items-center mb-3">
            <h4 class="card-title fw-bold text-white mb-0">Interactive Trajectory Map</h4>
            <div>
              <span class="badge bg-info text-dark me-2">Cyan = Raw GPS</span>
              <span class="badge bg-success me-2">Green = Ground Truth</span>
              <span class="badge bg-danger">Red = PathSnap Reconstructed</span>
            </div>
          </div>
          <iframe id="mapFrame" src="/visualize/00000081"></iframe>
        </div>
      </div>
    </div>
  </div>

  <script>
    function onSelectTrip() {{
      const sel = document.getElementById("tripSelect").value;
      document.getElementById("customTrip").value = sel;
    }}

    async function triggerReconstruction() {{
      let tid = document.getElementById("customTrip").value.trim() || document.getElementById("tripSelect").value;
      const beta = document.getElementById("paramBeta").value;
      const radius = document.getElementById("paramRadius").value;

      document.getElementById("loading").classList.remove("d-none");
      document.getElementById("runBtn").disabled = true;

      try {{
        const res = await fetch(`/reconstruct/${{tid}}?beta=${{beta}}&search_radius_m=${{radius}}`, {{ method: "POST" }});
        const data = await res.json();

        if (res.ok) {{
          // Update metric cards
          document.getElementById("metricsPanel").classList.remove("d-none");
          document.getElementById("valF1").innerText = data.metrics_hmm.route_f1.toFixed(3);
          document.getElementById("valPrecision").innerText = (data.metrics_hmm.route_precision * 100).toFixed(1) + "%";
          document.getElementById("valEditDist").innerText = data.metrics_hmm.normalized_edit_distance.toFixed(3);

          document.getElementById("tableHmmF1").innerText = data.metrics_hmm.route_f1.toFixed(4);
          document.getElementById("tableHmmDist").innerText = data.metrics_hmm.normalized_edit_distance.toFixed(4);

          document.getElementById("tableMlF1").innerText = data.metrics_ml_baseline.route_f1.toFixed(4);
          document.getElementById("tableMlDist").innerText = data.metrics_ml_baseline.normalized_edit_distance.toFixed(4);

          document.getElementById("tableGeoF1").innerText = data.metrics_geometric_baseline.route_f1.toFixed(4);
          document.getElementById("tableGeoDist").innerText = data.metrics_geometric_baseline.normalized_edit_distance.toFixed(4);

          document.getElementById("timeStats").innerText = `Observations: ${{data.n_observations}} | Time: ${{data.duration_s}}s | Candidates: ${{data.n_candidates_total}}`;

          // Refresh map iframe
          document.getElementById("mapFrame").src = data.map_url + "?t=" + new Date().getTime();
        }} else {{
          alert("Error: " + (data.detail || "Reconstruction failed."));
        }}
      }} catch (err) {{
        alert("Failed to connect to backend: " + err);
      }} finally {{
        document.getElementById("loading").classList.add("d-none");
        document.getElementById("runBtn").disabled = false;
      }}
    }}
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="PathSnap Distributed FastAPI Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind (0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    local_ip = get_local_ip()
    print("=" * 70)
    print("STARTING PATHSNAP DISTRIBUTED INFERENCE SERVER")
    print("=" * 70)
    print(f"Backend Server Bound To:  http://{args.host}:{args.port}")
    print(f"Laptop 1 Local Access:    http://localhost:{args.port}")
    print(f"Laptop 2 Network Access:  http://{local_ip}:{args.port}")
    print(f"Swagger API Docs:         http://{local_ip}:{args.port}/docs")
    print("=" * 70)
    print("Keep this terminal open on Laptop 1 during evaluation.")
    print("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
