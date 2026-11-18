"""
src/data_loader.py
==================
Modular data loader for PathSnap: Distributed Map-Matching & Route Reconstruction.

Provides functions to discover, parse, and load raw trip files:
- .track : GPS observations [longitude, latitude, timestamp]
- .nodes : Road network nodes [longitude, latitude] (0-based row index = node_id)
- .arcs  : Road network directed edges [from_node_id, to_node_id] (0-based row index = arc_id)
- .route : Sequence of ground-truth arc IDs
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Default base path relative to project root
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FALLBACK_DATA_DIR = Path(__file__).resolve().parent.parent / "Student folder" / "Student folder"


def get_data_dir(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and verify the dataset root directory."""
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            return p
    if DEFAULT_DATA_DIR.exists():
        return DEFAULT_DATA_DIR
    if FALLBACK_DATA_DIR.exists():
        return FALLBACK_DATA_DIR
    raise FileNotFoundError(f"Could not find dataset directory at {DEFAULT_DATA_DIR} or {FALLBACK_DATA_DIR}")


def get_available_partitions(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return available partition subdirectories (e.g. ['train', 'student_test'])."""
    base = get_data_dir(data_dir)
    partitions = []
    for item in sorted(os.listdir(base)):
        item_path = base / item
        if item_path.is_dir() and not item.startswith("."):
            partitions.append(item)
    return partitions


def get_trip_ids(partition: str = "train", data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """Return sorted list of trip folder IDs for a given partition."""
    base = get_data_dir(data_dir) / partition
    if not base.exists():
        raise FileNotFoundError(f"Partition directory not found: {base}")
    trip_ids = [
        d for d in sorted(os.listdir(base))
        if (base / d).is_dir() and not d.startswith(".")
    ]
    return trip_ids


def get_trip_file_paths(trip_id: str, partition: str = "train", data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """Return paths to .track, .nodes, .arcs, .route for a trip."""
    trip_dir = get_data_dir(data_dir) / partition / trip_id
    return {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "route": trip_dir / f"{trip_id}.route",
    }


def load_track(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .track file containing GPS observations.
    Columns: longitude (float), latitude (float), timestamp (float).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
        dtype={"longitude": np.float64, "latitude": np.float64, "timestamp": np.float64},
    )
    return df


def load_nodes(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .nodes file containing road-network node coordinates.
    0-based row index is node_id.
    Columns: longitude (float), latitude (float). Index: node_id (int).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Nodes file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
        dtype={"longitude": np.float64, "latitude": np.float64},
    )
    df.index.name = "node_id"
    return df


def load_arcs(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load .arcs file containing directed road-network edges.
    0-based row index is arc_id.
    Columns: from_node_id (int), to_node_id (int). Index: arc_id (int).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Arcs file not found: {path}")
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype={"from_node_id": np.int64, "to_node_id": np.int64},
    )
    df.index.name = "arc_id"
    return df


def load_route(file_path: Union[str, Path]) -> pd.Series:
    """
    Load .route file containing chronological sequence of reference arc_id values.
    Returns: pd.Series of arc_ids.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Route file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    s = df.iloc[:, 0]
    s.name = "arc_id"
    return s.astype(np.int64)


def load_trip(trip_id: str, partition: str = "train", data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Load all four data components for a trip."""
    paths = get_trip_file_paths(trip_id, partition, data_dir)
    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": load_track(paths["track"]),
        "nodes": load_nodes(paths["nodes"]),
        "arcs": load_arcs(paths["arcs"]),
        "route": load_route(paths["route"]),
    }


if __name__ == "__main__":
    partitions = get_available_partitions()
    print(f"Available partitions: {partitions}")
    for part in partitions:
        trips = get_trip_ids(part)
        print(f"Partition '{part}': {len(trips)} trips found.")
        if trips:
            sample = load_trip(trips[0], part)
            print(f"  Sample trip {trips[0]}:")
            print(f"    Track shape: {sample['track'].shape}")
            print(f"    Nodes shape: {sample['nodes'].shape}")
            print(f"    Arcs shape:  {sample['arcs'].shape}")
            print(f"    Route length: {len(sample['route'])}")
