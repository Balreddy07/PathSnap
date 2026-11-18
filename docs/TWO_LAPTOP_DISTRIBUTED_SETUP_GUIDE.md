# PathSnap: Two-Laptop Distributed Setup & Evaluation Guide

> **Project:** PathSnap – Distributed Map-Matching & Route Reconstruction  
> **Course:** UE24CS352A - Machine Learning (Semester 5)  
> **Architecture:** Distributed Client-Server over Local Area Network (LAN)  
> **Date:** November 18, 2026

---

## 1. System Roles Overview

The PathSnap distributed architecture separates heavy model inference and graph computations from the interactive evaluator interface:

```
┌────────────────────────────────────────────────────────┐
│             LAPTOP 1: BACKEND & INFERENCE ENGINE       │
│                                                        │
│  - Trained XGBoost & Random Forest Models              │
│  - Road Network MultiDiGraph Topology                  │
│  - HMM Emission & Transition Probability Calculators   │
│  - Viterbi Dynamic Programming Route Decoder           │
│  - FastAPI REST API Server (Port 8000)                 │
└───────────────────────────┬────────────────────────────┘
                            │
               Local Wi-Fi / Hotspot / LAN
             (e.g., http://10.20.204.228:8000)
                            │
┌───────────────────────────▼────────────────────────────┐
│             LAPTOP 2: EVALUATOR / CLIENT INTERFACE     │
│                                                        │
│  - Mode A: Web Browser UI (Interactive Dashboard)      │
│  - Mode B: Python Evaluator Client (`client.py`)       │
│  - Mode C: Swagger API Docs (`/docs`)                  │
│  - Live Leaflet / Folium Map Rendering                │
│  - Real-time Metric Comparison against Ground Truth    │
└────────────────────────────────────────────────────────┘
```

---

## 2. Step-by-Step Setup Instructions

### Step 1: Connect Both Laptops to the Same Network
* Connect both **Laptop 1** and **Laptop 2** to the **same Wi-Fi network** (or create a **Mobile Hotspot** from a phone and connect both laptops to it).

---

### Step 2: Configure & Start Laptop 1 (Inference Server)

1. Open PowerShell or Command Prompt on **Laptop 1**.
2. Navigate to the project directory:
   ```powershell
   cd D:\SEM5\ML
   ```
3. Activate the virtual environment:
   ```powershell
   .venv\Scripts\activate
   ```
4. Find Laptop 1's IP address:
   ```powershell
   ipconfig
   ```
   *Look for `IPv4 Address` under your active Wi-Fi or Wireless LAN adapter (e.g., `10.20.204.228` or `192.168.1.15`).*

5. Start the PathSnap FastAPI server:
   ```powershell
   python src/api/server.py --host 0.0.0.0 --port 8000
   ```
   *You will see:*
   ```text
   ======================================================================
   STARTING PATHSNAP DISTRIBUTED INFERENCE SERVER
   ======================================================================
   Backend Server Bound To:  http://0.0.0.0:8000
   Laptop 1 Local Access:    http://localhost:8000
   Laptop 2 Network Access:  http://10.20.204.228:8000
   Swagger API Docs:         http://10.20.204.228:8000/docs
   ======================================================================
   ```
   **Leave this terminal running.**

---

### Step 3: Connect & Evaluate from Laptop 2 (Client / Evaluator)

On **Laptop 2**, you have **three options** to interact with the system:

#### Option A: Interactive Web Dashboard (Recommended for Evaluation & Demo)
1. Open any web browser (Chrome, Edge, Firefox, Safari).
2. Enter Laptop 1's IP address in the address bar:
   ```text
   http://<LAPTOP_1_IP>:8000
   ```
   *(e.g., `http://10.20.204.228:8000`)*
3. **What You Can Do on the Dashboard:**
   * **Select a Trip:** Pick any validation trip (e.g., `00000081`, `00000005`, `00000004`) from the dropdown.
   * **Adjust Parameters:** Modify HMM $\beta$ parameter (default: 20m) and search radius (default: 50m).
   * **Click "Run Map-Matching & Reconstruct":** Sends a real-time POST request to Laptop 1.
   * **View Metric Cards:** Instantly view Route F1-Score, Precision, and Normalized Edit Distance comparing:
     - PathSnap HMM (Green badge)
     - ML-Only Snapping (Purple badge)
     - Geometric Nearest Snapping (Yellow badge)
   * **Interactive Map:** Pan, zoom, and toggle layers (Cyan = Raw GPS, Green = Ground Truth, Red = PathSnap Reconstructed Route).

---

#### Option B: Terminal Client (`client.py`)
If Laptop 2 has Python installed and the PathSnap code cloned:
```bash
# Check connectivity:
python src/api/client.py --server http://<LAPTOP_1_IP>:8000 --trips

# Reconstruct a specific trip:
python src/api/client.py --server http://<LAPTOP_1_IP>:8000 --trip-id 00000081

# Run batch comparative evaluation:
python src/api/client.py --server http://<LAPTOP_1_IP>:8000 --batch
```

---

#### Option C: Swagger API Interactive Documentation
1. In Laptop 2's browser, open:
   ```text
   http://<LAPTOP_1_IP>:8000/docs
   ```
2. Test any endpoint live (`/health`, `/trips`, `/reconstruct/{trip_id}`, `/visualize/{trip_id}`) directly from the interactive OpenAPI UI.

---

## 3. Firewall & Troubleshooting (Windows)

If Laptop 2 cannot connect to `http://<LAPTOP_1_IP>:8000`:

1. **Test Ping from Laptop 2:**
   Open terminal on Laptop 2 and run:
   ```cmd
   ping <LAPTOP_1_IP>
   ```
2. **Windows Firewall Rule on Laptop 1 (Allow Port 8000):**
   Run PowerShell as Administrator on Laptop 1:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap Server" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
   *(Or temporarily set your Wi-Fi network profile from "Public" to "Private" in Windows Settings).*
3. **Ensure Both Devices are on the Same Subnet:**
   Both IP addresses should start with the same prefix (e.g. `192.168.1.X` or `10.20.204.X`). If on university Wi-Fi with client isolation, use a **Mobile Hotspot**.

---

## 4. Live Demonstration Script for Evaluators

When presenting to professors or evaluators:

1. **Show Laptop 1 (Backend):** Point to the terminal running `server.py`, showing active logging of incoming requests, candidate emission calculations, and Viterbi decoding times.
2. **Show Laptop 2 (Dashboard):**
   * Load `http://<LAPTOP_1_IP>:8000`.
   * Select **Trip `00000005`** and click Run: show how PathSnap achieves **0.9073 F1** and **93.0% Precision**.
   * Select **Trip `00000081`** (complex 3,224-point route): show how the HMM bridges disconnected gaps and maintains strict road network topology where geometric snapping fails.
