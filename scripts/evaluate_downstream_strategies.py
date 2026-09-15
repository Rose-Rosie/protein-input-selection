from project_paths import ROOT
import argparse
import json
import math
import time
from itertools import product
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_split(task: str):
    split = np.load(ROOT / 'splits' / f'{task}_split_entries.npz', allow_pickle=True)
    return {'kind': 'multilabel', 'train_entries': split['train_entries'].astype(str), 'val_entries': split['val_entries'].astype(str), 'test_entries': split['test_entries'].astype(str), 'y_train': split['y_train'].astype(int), 'y_val': split['y_val'].astype(int), 'y_test': split['y_test'].astype(int), 'labels': split['label_cols'].astype(str).tolist()}

def load_embeddings_in_order(path: Path, entries: np.ndarray) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    missing = [entry for entry in entries if entry not in data.files]
    if missing:
        raise KeyError(f'{path} missing {len(missing)} entries; first missing={missing[:5]}')
    x = np.stack([np.asarray(data[entry], dtype=np.float32) for entry in entries], axis=0)
    if x.ndim != 2:
        raise ValueError(f'{path} produced non-2D matrix: {x.shape}')
    finite = np.isfinite(x)
    if not finite.all():
        bad = np.argwhere(~finite)
        raise ValueError(f'{path} contains NaN/Inf at {bad[:5].tolist()} total_bad={bad.shape[0]}')
    return x

def probs_from_multioutput(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    cols = []
    for prob in probs:
        if prob.shape[1] == 1:
            cols.append(np.zeros(prob.shape[0], dtype=np.float32))
        else:
            cols.append(prob[:, 1])
    return np.stack(cols, axis=1)

class RobustBinaryMultiOutput:

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.constants_ = []

    def fit(self, x, y):
        self.estimators_ = []
        self.constants_ = []
        for j in range(y.shape[1]):
            yj = y[:, j]
            classes = np.unique(yj)
            if len(classes) < 2:
                self.estimators_.append(None)
                self.constants_.append(int(classes[0]))
            else:
                est = clone(self.base_estimator)
                est.fit(x, yj)
                self.estimators_.append(est)
                self.constants_.append(None)
        return self

    def predict(self, x):
        cols = []
        for est, const in zip(self.estimators_, self.constants_):
            if est is None:
                cols.append(np.full(x.shape[0], const, dtype=int))
            else:
                cols.append(est.predict(x).astype(int))
        return np.stack(cols, axis=1)

    def predict_proba(self, x):
        probs = []
        for est, const in zip(self.estimators_, self.constants_):
            if est is None:
                p1 = np.full(x.shape[0], float(const), dtype=np.float32)
                probs.append(np.stack([1.0 - p1, p1], axis=1))
            else:
                prob = est.predict_proba(x)
                if prob.shape[1] == 1:
                    cls = int(est.classes_[0])
                    p1 = np.full(x.shape[0], float(cls), dtype=np.float32)
                    prob = np.stack([1.0 - p1, p1], axis=1)
                elif list(est.classes_) == [1]:
                    p1 = prob[:, 0]
                    prob = np.stack([1.0 - p1, p1], axis=1)
                elif list(est.classes_) == [0]:
                    p1 = np.zeros(x.shape[0], dtype=np.float32)
                    prob = np.stack([1.0 - p1, p1], axis=1)
                probs.append(prob)
        return probs

def evaluate_multilabel(y_true, y_pred, y_prob) -> dict:
    metrics = {'subset_accuracy': accuracy_score(y_true, y_pred)}
    for avg in ['micro', 'macro', 'weighted']:
        metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    try:
        metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
    except Exception:
        metrics['roc_auc_macro'] = np.nan
    try:
        metrics['average_precision_macro'] = average_precision_score(y_true, y_prob, average='macro')
    except Exception:
        metrics['average_precision_macro'] = np.nan
    return metrics

def select_multilabel_thresholds(y_val, prob_val):
    thresholds = []
    for j in range(y_val.shape[1]):
        best_t, best_f1 = (0.5, -1.0)
        for t in np.linspace(0.1, 0.9, 17):
            pred = (prob_val[:, j] >= t).astype(int)
            score = f1_score(y_val[:, j], pred, zero_division=0)
            if score > best_f1:
                best_t, best_f1 = (float(t), float(score))
        thresholds.append(best_t)
    return np.asarray(thresholds, dtype=np.float32)

def tune_lasso(kind, x_train, y_train, x_val, y_val, seed):
    c_values = [0.01, 0.1, 1.0, 5.0, 10.0]
    best = None
    for c in c_values:
        base = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=1000, random_state=seed))
        model = RobustBinaryMultiOutput(base)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = f1_score(y_val, pred, average='macro', zero_division=0)
        if best is None or score > best['score']:
            best = {'C': c, 'score': float(score)}
    return best

def fit_lasso(kind, x_train, y_train, x_val, y_val, best, seed):
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val], axis=0)
    base = make_pipeline(StandardScaler(), LogisticRegression(penalty='l1', solver='liblinear', C=best['C'], max_iter=1000, random_state=seed))
    model = RobustBinaryMultiOutput(base)
    model.fit(x_combined, y_combined)
    return model

def xgb_param_candidates(seed: int, n_trials: int):
    grid = list(product([100, 200, 300, 500], [0.01, 0.03, 0.05, 0.1], [4, 6, 8], [0.7, 0.8, 1.0], [0.7, 0.8, 1.0]))
    rng = np.random.default_rng(seed)
    rng.shuffle(grid)
    for n_estimators, lr, depth, subsample, colsample in grid[:n_trials]:
        yield {'n_estimators': n_estimators, 'learning_rate': lr, 'max_depth': depth, 'subsample': subsample, 'colsample_bytree': colsample}

def make_xgb(kind, params, seed, num_workers):
    if XGBClassifier is None:
        raise ModuleNotFoundError("xgboost is required for downstream_model='xgboost'")
    objective = 'binary:logistic'
    eval_metric = 'logloss'
    return XGBClassifier(**params, objective=objective, eval_metric=eval_metric, random_state=seed, n_jobs=num_workers, tree_method='hist', device='cuda')

def tune_xgboost(kind, x_train, y_train, x_val, y_val, seed, n_trials, num_workers):
    best = None
    for params in xgb_param_candidates(seed, n_trials):
        base = make_xgb(kind, params, seed, num_workers)
        model = RobustBinaryMultiOutput(base)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = f1_score(y_val, pred, average='macro', zero_division=0)
        if best is None or score > best['score']:
            best = {'params': params, 'score': float(score)}
    return best

def fit_xgboost(kind, x_train, y_train, x_val, y_val, best, seed, num_workers):
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val], axis=0)
    base = make_xgb(kind, best['params'], seed, num_workers)
    model = RobustBinaryMultiOutput(base)
    model.fit(x_combined, y_combined)
    return model

class MLP(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden: list[int], dropout: float, kind: str):
        super().__init__()
        layers = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.kind = kind

    def forward(self, x):
        return self.net(x)

def mlp_candidates(seed: int, n_trials: int):
    base = [{'hidden': [256], 'dropout': 0.2, 'lr': 0.001}, {'hidden': [512], 'dropout': 0.2, 'lr': 0.001}, {'hidden': [512, 256], 'dropout': 0.3, 'lr': 0.001}, {'hidden': [1024, 512], 'dropout': 0.3, 'lr': 0.0005}, {'hidden': [512, 512, 256], 'dropout': 0.4, 'lr': 0.0005}, {'hidden': [256, 128], 'dropout': 0.2, 'lr': 0.005}, {'hidden': [1024], 'dropout': 0.4, 'lr': 0.001}, {'hidden': [768, 384], 'dropout': 0.3, 'lr': 0.001}, {'hidden': [384], 'dropout': 0.2, 'lr': 0.0005}, {'hidden': [640, 320], 'dropout': 0.4, 'lr': 0.0005}, {'hidden': [128], 'dropout': 0.2, 'lr': 0.01}, {'hidden': [768, 512, 256], 'dropout': 0.3, 'lr': 0.0005}]
    rng = np.random.default_rng(seed)
    order = np.arange(len(base))
    rng.shuffle(order)
    for idx in order[:n_trials]:
        yield base[int(idx)]

def standardize_arrays(x_train, x_val, x_test):
    scaler = StandardScaler()
    return (scaler.fit_transform(x_train).astype(np.float32), scaler.transform(x_val).astype(np.float32), scaler.transform(x_test).astype(np.float32))

def train_one_mlp(kind, params, x_train, y_train, x_val, y_val, seed, max_epochs, patience, batch_size):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = y_train.shape[1]
    model = MLP(x_train.shape[1], output_dim, params['hidden'], params['dropout'], kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = nn.BCEWithLogitsLoss()
    y_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32), y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    best_state, best_score, best_epoch = (None, -math.inf, 0)
    no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = (xb.to(device), yb.to(device))
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        prob_val = predict_mlp_probs(model, kind, x_val)
        thresh = select_multilabel_thresholds(y_val, prob_val)
        pred_val = (prob_val >= thresh).astype(int)
        score = f1_score(y_val, pred_val, average='macro', zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return (model, {'params': params, 'score': best_score, 'best_epoch': best_epoch})

def predict_mlp_probs(model, kind, x):
    device = next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            xb = torch.as_tensor(x[start:start + 4096], dtype=torch.float32, device=device)
            logits = model(xb)
            out = torch.sigmoid(logits)
            probs.append(out.detach().cpu().numpy())
    return np.vstack(probs)

def tune_mlp(kind, x_train, y_train, x_val, y_val, seed, n_trials, max_epochs, patience, batch_size):
    best_model, best = (None, None)
    for i, params in enumerate(mlp_candidates(seed, n_trials)):
        model, info = train_one_mlp(kind, params, x_train, y_train, x_val, y_val, seed + i, max_epochs, patience, batch_size)
        if best is None or info['score'] > best['score']:
            best_model, best = (model, info)
    return (best_model, best)

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)

STRATEGIES_1000AA = ['head1000', 'mid1000', 'tail1000', 'splice300_400_300', 'domain_center_longest', 'domain_max_cover']

from xgboost import XGBClassifier
