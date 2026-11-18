"""tests/test_phase3_features.py"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
import pytest, numpy as np
from src.models.data_loader import load_config, load_datasets, add_binary_target
from src.models.feature_preprocessing import get_feature_list, prepare_X_y, fit_scaler, transform

def test_all_features_present():
    cfg = load_config()
    train, val = load_datasets(cfg)
    features = get_feature_list(cfg)
    for f in features:
        assert f in train.columns, f"Missing feature in train: {f}"
        assert f in val.columns,   f"Missing feature in val: {f}"

def test_scaler_fits_on_train_only():
    cfg = load_config()
    train, val = load_datasets(cfg)
    train = add_binary_target(train)
    val   = add_binary_target(val)
    features = get_feature_list(cfg)
    X_tr, y_tr = prepare_X_y(train, features)
    X_va, y_va = prepare_X_y(val, features)
    pipe = fit_scaler(X_tr)
    X_tr_scaled = transform(pipe, X_tr)
    X_va_scaled = transform(pipe, X_va)
    assert X_tr_scaled.shape[1] == len(features)
    assert X_va_scaled.shape[1] == len(features)
    # Scaled training data should be approx zero-mean
    assert abs(X_tr_scaled.mean()) < 0.5

def test_no_inf_after_preprocessing():
    cfg = load_config()
    train, val = load_datasets(cfg)
    train = add_binary_target(train)
    features = get_feature_list(cfg)
    X_tr, _ = prepare_X_y(train, features)
    pipe = fit_scaler(X_tr)
    scaled = transform(pipe, X_tr)
    assert not np.any(np.isinf(scaled))
    assert not np.any(np.isnan(scaled))
