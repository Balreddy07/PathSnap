"""tests/test_phase3_no_leakage.py"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import pytest
from src.models.data_loader import load_config
from src.models.feature_preprocessing import get_feature_list

LEAKAGE_COLS = {"true_arc_id", "distance_to_true_arc", "label", "split",
                "trip_id", "gps_index", "timestamp", "gps_longitude", "gps_latitude",
                "candidate_arc_id"}

def test_no_leakage_columns_in_features():
    cfg = load_config()
    features = set(get_feature_list(cfg))
    leaking = features & LEAKAGE_COLS
    assert len(leaking) == 0, f"Leakage columns in feature set: {leaking}"

def test_binary_target_not_in_features():
    cfg = load_config()
    features = set(get_feature_list(cfg))
    assert "binary_target" not in features
