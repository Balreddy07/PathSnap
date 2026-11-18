"""
src/api/client.py
=================
Lightweight Evaluator Client for Laptop 2.
Connects over LAN to Laptop 1's PathSnap Inference Server.
Usage:
    python src/api/client.py --server http://192.168.1.100:8000 --trip-id 00000081
    python src/api/client.py --server http://192.168.1.100:8000 --trips
    python src/api/client.py --server http://192.168.1.100:8000 --batch
"""

import sys
import argparse
import json
import urllib.request
import urllib.error


def make_request(url: str, method: str = "GET", payload: dict = None) -> dict:
    req = urllib.request.Request(url, method=method)
    if payload:
        data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
        req.data = data
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection Failed: {e.reason}")
        print("Check if Laptop 1 is running `python src/api/server.py` and both laptops are on the same Wi-Fi.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PathSnap Laptop 2 Evaluator Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Laptop 1 Server URL (e.g. http://192.168.1.50:8000)")
    parser.add_argument("--trip-id", type=str, default=None, help="Trip ID to evaluate (e.g. 00000081)")
    parser.add_argument("--trips", action="store_true", help="List available trips on server")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation on sample validation trips")
    args = parser.parse_args()

    base_url = args.server.rstrip("/")

    # 1. Health check
    print(f"Connecting to PathSnap Backend at {base_url}...")
    health = make_request(f"{base_url}/health")
    print(f"Server Status: {health.get('status')} | Service: {health.get('service')}\n")

    if args.trips:
        trips_info = make_request(f"{base_url}/trips")
        print(f"Total Available Trips: {trips_info.get('total_trips')}")
        print("Recommended Validation Trips:", trips_info.get("validation_recommended"))
        return

    trip_list = [args.trip_id] if args.trip_id else (["00000005", "00000081", "00000004"] if args.batch else ["00000081"])

    print("=" * 80)
    print(f"RUNNING MAP-MATCHING EVALUATION ON {len(trip_list)} TRIP(S)")
    print("=" * 80)

    for tid in trip_list:
        print(f"\n[Trip {tid}] Requesting reconstruction from Laptop 1...")
        res = make_request(f"{base_url}/reconstruct/{tid}", method="POST")

        h = res["metrics_hmm"]
        m = res["metrics_ml_baseline"]
        g = res["metrics_geometric_baseline"]

        print(f"  Time Taken: {res['duration_s']}s | GPS Fixes: {res['n_observations']} | Candidates: {res['n_candidates_total']}")
        print(f"  PathSnap HMM:       F1={h['route_f1']:.4f} | Precision={h['route_precision']*100:.1f}% | NormEditDist={h['normalized_edit_distance']:.4f}")
        print(f"  ML-Only Baseline:   F1={m['route_f1']:.4f} | Recall={m['route_recall']*100:.1f}%    | NormEditDist={m['normalized_edit_distance']:.4f}")
        print(f"  Geometric Baseline: F1={g['route_f1']:.4f} | Recall={g['route_recall']*100:.1f}%    | NormEditDist={g['normalized_edit_distance']:.4f}")
        print(f"  Interactive Map URL: {base_url}{res['map_url']}")


if __name__ == "__main__":
    main()
