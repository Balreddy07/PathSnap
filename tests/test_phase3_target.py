"""tests/test_phase3_target.py"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import pytest
from src.models.data_loader import load_config, load_datasets, add_binary_target

def test_target_binary_values():
    cfg = load_config()
    train, val = load_datasets(cfg)
    for name, df in [("train", train), ("val", val)]:
        df = add_binary_target(df)
        vals = set(df["binary_target"].unique())
        assert vals.issubset({0, 1}), f"{name}: unexpected target values {vals}"

def test_candidate_arc_matches_true_arc_gives_positive():
    cfg = load_config()
    train, _ = load_datasets(cfg)
    train = add_binary_target(train)
    match_rows = train[train["candidate_arc_id"] == train["true_arc_id"]]
    assert (match_rows["binary_target"] == 1).all()

def test_nonmatch_gives_negative():
    cfg = load_config()
    train, _ = load_datasets(cfg)
    train = add_binary_target(train)
    nonmatch = train[train["candidate_arc_id"] != train["true_arc_id"]]
    assert (nonmatch["binary_target"] == 0).all()
