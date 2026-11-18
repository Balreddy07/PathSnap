"""
src/models/feature_preprocessing.py
=====================================
Shared feature preprocessing for all five Phase 3 models.
Fits transformations on training data only; applies to validation data.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


BOOL_FEATURES = ["is_outlier", "is_stationary"]


def get_feature_list(config: Dict[str, Any]) -> List[str]:
    return config["features"]["predictive"]


def _replace_inf(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    return df


def _cast_bools(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BOOL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def prepare_X_y(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "binary_target",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y from dataframe."""
    df = _cast_bools(df)
    df = _replace_inf(df, features)
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  WARNING: missing columns {missing}")
    X = df[present].copy()
    y = df[target_col].copy()
    return X, y


def build_scaler_pipeline() -> Pipeline:
    """Imputer + StandardScaler pipeline for LR, SVM, MLP."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def fit_scaler(X_train: pd.DataFrame) -> Pipeline:
    pipe = build_scaler_pipeline()
    pipe.fit(X_train)
    return pipe


def transform(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipe.transform(X)


def save_scaler(pipe: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Scaler saved: {path}")


def load_scaler(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def impute_only(X: pd.DataFrame) -> np.ndarray:
    """For tree models: impute NaNs but do NOT scale."""
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)


def prepare_all(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Returns X_train, X_val, y_train, y_val, scaler_pipeline (or None).
    scale=True  → StandardScaler (for LR, SVM, MLP)
    scale=False → median impute only (for RF, XGBoost)
    """
    features   = get_feature_list(config)
    target_col = config["target"]["column"]

    X_train_df, y_train = prepare_X_y(train_df, features, target_col)
    X_val_df,   y_val   = prepare_X_y(val_df,   features, target_col)

    if scale:
        pipe    = fit_scaler(X_train_df)
        X_train = transform(pipe, X_train_df)
        X_val   = transform(pipe, X_val_df)
        return X_train, X_val, y_train.values, y_val.values, pipe
    else:
        imp     = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(X_train_df)
        X_val   = imp.transform(X_val_df)
        return X_train, X_val, y_train.values, y_val.values, imp
