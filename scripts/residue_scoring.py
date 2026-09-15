from project_paths import ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import evaluate_downstream_strategies as base

def load_matrix(path: Path, entries: np.ndarray) -> np.ndarray:
    return base.load_embeddings_in_order(path, entries)

def strategy_path(model: str, task: str, strategy: str) -> Path:
    if strategy == 'full_length_mean':
        return ROOT / 'embeddings_full_length' / model / task / 'full_length_mean.npz'
    if strategy == 'domain_only_mean':
        return ROOT / 'embeddings_full_length' / model / task / 'full_domain_pooling.npz'
    if strategy in {'head1000', 'domain_center_longest'}:
        return ROOT / 'embeddings_1000aa' / model / task / f'{strategy}.npz'
    if strategy.startswith('domain_matched_random_region_only_seed'):
        return ROOT / 'embeddings_region_ablation' / model / task / f'{strategy}.npz'
    raise ValueError(strategy)

def effective_coefficients(kind: str, final_model, outer: StandardScaler, dim: int) -> np.ndarray:
    outer_scale = np.where(outer.scale_ == 0, 1.0, outer.scale_).astype(np.float64)
    rows = []
    for est, const in zip(final_model.estimators_, final_model.constants_):
        if est is None:
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        inner = est.named_steps['standardscaler']
        clf = est.named_steps['logisticregression']
        inner_scale = np.where(inner.scale_ == 0, 1.0, inner.scale_).astype(np.float64)
        rows.append((clf.coef_[0].astype(np.float64) / inner_scale / outer_scale).astype(np.float32))
    return np.stack(rows, axis=0)

def task_global_signal(coeff: np.ndarray, residues: np.ndarray) -> np.ndarray:
    contributions = residues.astype(np.float32) @ coeff.T.astype(np.float32)
    return np.sqrt(np.mean(np.square(contributions, dtype=np.float64), axis=1)).astype(np.float32)
