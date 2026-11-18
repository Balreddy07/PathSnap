# PathSnap Phase 4 – Hyperparameter Optimization Technical Report

> **Project:** PathSnap – Distributed Map-Matching & Route Reconstruction
> **Phase:** Phase 4 – Hyperparameter Optimization (Optuna GroupKFold)
> **Date:** November 18, 2026

---

## 1. Objective
Phase 4 optimizes the two highest-performing candidate classification models from Phase 3: **XGBoost** and **Random Forest**.
The objective is to maximize candidate classification **F1-score** and **PR-AUC** while ensuring strict trip-level generalization through **GroupKFold cross-validation** (so GPS observations from the same trip never leak across folds).

## 2. Optimization Methodology
- **Framework:** Optuna with Tree-structured Parzen Estimator (TPE) Bayesian search.
- **Validation Strategy:** GroupKFold on training trips (64 trips). Final held-out validation set (`val.csv`, 16 trips) was left untouched until final model verification.
- **Evaluation Metric:** Macro F1-score & PR-AUC.

## 3. Best Hyperparameters Identified

### XGBoost Best Parameters
```json
{
  "n_estimators": 250,
  "max_depth": 7,
  "learning_rate": 0.081989299779174,
  "min_child_weight": 1,
  "subsample": 0.8078557723687835,
  "colsample_bytree": 0.754046027026078,
  "gamma": 0.35562388134591194,
  "reg_alpha": 0.4950712940314879,
  "reg_lambda": 0.19406786166905346
}
```

### Random Forest Best Parameters
```json
{
  "n_estimators": 120,
  "max_depth": 22,
  "min_samples_split": 7,
  "min_samples_leaf": 3,
  "max_features": "sqrt"
}
```

## 4. Before-vs-After Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest (Baseline Phase 3) | 0.8812 | 0.5779 | 0.8238 | 0.6793 | 0.9353 | 0.7576 | 93.5000 |
| RandomForest (Optimized Phase 4) | 0.8947 | 0.6251 | 0.7759 | 0.6924 | 0.9332 | 0.7526 | 200.2400 |
| XGBoost (Baseline Phase 3) | 0.8739 | 0.5575 | 0.8470 | 0.6724 | 0.9345 | 0.7584 | 14.6600 |
| XGBoost (Optimized Phase 4) | 0.8770 | 0.5667 | 0.8269 | 0.6725 | 0.9310 | 0.7505 | 53.8000 |

## 5. Selection for Phase 5 HMM / Viterbi Route Reconstruction
**Selected Emission Model:** **XGBoost**

### Justification:
1. **High Recall & Precision Balance:** A high recall on candidates ensures the true road arc is retained within the HMM state lattice at time $t$. Discarding the true arc at the emission stage makes correct Viterbi path recovery impossible.
2. **Inference Latency:** XGBoost evaluates candidate probabilities in **0.28 seconds for 195k candidates** (approx 700k candidates/sec), making real-time map matching computationally viable.
3. **Calibrated Likelihood:** XGBoost logistic loss output naturally maps to observation likelihoods $P(O_t \mid S_t)$ for HMM emission modeling.

---
*Report generated automatically by src/optimization/run_phase4.py*