"""tests/test_phase3_data_loading.py"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import pytest, pandas as pd
from src.models.data_loader import load_config, load_datasets, add_binary_target, validate_split

def test_config_loads():
    cfg = load_config()
    assert "data" in cfg
    assert "features" in cfg
    assert "models" in cfg

def test_csvs_exist():
    cfg = load_config()
    assert os.path.exists(os.path.join(PROJECT_ROOT, cfg["data"]["train_csv"]))
    assert os.path.exists(os.path.join(PROJECT_ROOT, cfg["data"]["val_csv"]))

def test_datasets_load():
    cfg = load_config()
    train, val = load_datasets(cfg)
    assert len(train) > 0
    assert len(val) > 0

def test_no_trip_overlap():
    cfg = load_config()
    train, val = load_datasets(cfg)
    info = validate_split(train, val)
    assert info["overlap_count"] == 0, f"Trip overlap found: {info['overlap']}"

def test_binary_target_derived():
    cfg = load_config()
    train, val = load_datasets(cfg)
    train = add_binary_target(train)
    val   = add_binary_target(val)
    assert "binary_target" in train.columns
    assert set(train["binary_target"].unique()).issubset({0, 1})
    assert set(val["binary_target"].unique()).issubset({0, 1})

def test_positive_rate_nonzero():
    cfg = load_config()
    train, _ = load_datasets(cfg)
    train = add_binary_target(train)
    pct_pos = (train["binary_target"] == 1).mean() * 100
    assert pct_pos > 0, "Zero positive labels – check GPS column order in generate_dataset.py"
