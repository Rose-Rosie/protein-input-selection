import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import evaluate_downstream_strategies as base
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
MODELS = ['prott5', 'esm2']
STRATEGIES = ['full_length_mean', 'full_domain_pooling'] + base.STRATEGIES_1000AA + [f'domain_matched_random_region_only_seed{s}' for s in [101,102,103,104,105]]
DOWNSTREAM_MODELS = ['lasso', 'xgboost', 'dnn']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate independently encoded truncation representations and export paired protein-level outcomes.')
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS)
    parser.add_argument('--embedding-models', nargs='+', default=MODELS, choices=MODELS)
    parser.add_argument('--strategies', nargs='+', default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument('--downstream-models', nargs='+', default=DOWNSTREAM_MODELS, choices=DOWNSTREAM_MODELS)
    parser.add_argument('--outdir', default=str(ROOT / 'downstream_results'))
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--xgb-trials', type=int, default=12)
    parser.add_argument('--dnn-trials', type=int, default=12)
    parser.add_argument('--dnn-max-epochs', type=int, default=50)
    parser.add_argument('--dnn-patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--max-samples-per-split', type=int, default=None)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    return parser.parse_args()

def embedding_path(model, task, strategy):
    if strategy in {'full_length_mean', 'full_domain_pooling'}:
        return ROOT / 'embeddings_full_length' / model / task / f'{strategy}.npz'
    if strategy.startswith('domain_matched_random_region_only_seed'):
        return ROOT / 'embeddings_region_ablation' / model / task / f'{strategy}.npz'
    return ROOT / 'embeddings_1000aa' / model / task / f'{strategy}.npz'


def take_indices(n: int, max_n: Optional[int], seed: int) -> np.ndarray:
    if max_n is None or n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))

def slice_split(split: dict, max_n: Optional[int], seed: int) -> dict:
    result = dict(split)
    for offset, name in enumerate(['train', 'val', 'test']):
        idx = take_indices(len(split[f'{name}_entries']), max_n, seed + offset)
        result[f'{name}_entries'] = split[f'{name}_entries'][idx]
        result[f'y_{name}'] = split[f'y_{name}'][idx]
    return result

def choose_lasso(kind, x_train, y_train, x_val, y_val, seed):
    best = base.tune_lasso(kind, x_train, y_train, x_val, y_val, seed)
    tuned_model = base.fit_lasso(kind, x_train, y_train, np.empty((0, x_train.shape[1])), y_train[:0], best, seed)
    val_prob = base.probs_from_multioutput(tuned_model, x_val)
    thresholds = base.select_multilabel_thresholds(y_val, val_prob)
    final_model = base.fit_lasso(kind, x_train, y_train, x_val, y_val, best, seed)
    return (final_model, best, thresholds)

def choose_xgboost(kind, x_train, y_train, x_val, y_val, seed, n_trials, num_workers):
    best = base.tune_xgboost(kind, x_train, y_train, x_val, y_val, seed, n_trials, num_workers)
    empty_x = np.empty((0, x_train.shape[1]), dtype=x_train.dtype)
    tuned_model = base.fit_xgboost(kind, x_train, y_train, empty_x, y_train[:0], best, seed, num_workers)
    val_prob = base.probs_from_multioutput(tuned_model, x_val)
    thresholds = base.select_multilabel_thresholds(y_val, val_prob)
    final_model = base.fit_xgboost(kind, x_train, y_train, x_val, y_val, best, seed, num_workers)
    return (final_model, best, thresholds)

def train_mlp_fixed_epochs(kind, params, x, y, seed, epochs, batch_size):
    base.set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dim = y.shape[1]
    model = base.MLP(x.shape[1], output_dim, params['hidden'], params['dropout'], kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = nn.BCEWithLogitsLoss()
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), y_tensor)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, yb in loader:
            xb, yb = (xb.to(device), yb.to(device))
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model

def choose_dnn(kind, x_train, y_train, x_val, y_val, seed, args):
    train_scaler = StandardScaler().fit(x_train)
    x_train_tune = train_scaler.transform(x_train).astype(np.float32)
    x_val_tune = train_scaler.transform(x_val).astype(np.float32)
    tuned_model, best = base.tune_mlp(kind, x_train_tune, y_train, x_val_tune, y_val, seed, args.dnn_trials, args.dnn_max_epochs, args.dnn_patience, args.batch_size)
    val_prob = base.predict_mlp_probs(tuned_model, kind, x_val_tune)
    thresholds = base.select_multilabel_thresholds(y_val, val_prob)
    x_combined = np.vstack([x_train, x_val])
    y_combined = np.concatenate([y_train, y_val], axis=0)
    final_scaler = StandardScaler().fit(x_combined)
    x_combined_s = final_scaler.transform(x_combined).astype(np.float32)
    final_model = train_mlp_fixed_epochs(kind, best['params'], x_combined_s, y_combined, seed + 1000, best['best_epoch'], args.batch_size)
    best['final_epochs'] = int(best['best_epoch'])
    return (final_model, final_scaler, best, thresholds)

def predict_model(kind, downstream, model, x, thresholds, scaler=None):
    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)
    if downstream == 'dnn':
        prob = base.predict_mlp_probs(model, kind, x)
    else:
        prob = base.probs_from_multioutput(model, x)
    pred = (prob >= thresholds).astype(int)
    return (pred, prob)

def multilabel_protein_rows(entries, y_true, y_pred, y_prob):
    eps = 1e-07
    prob = np.clip(y_prob.astype(np.float64), eps, 1.0 - eps)
    signed_margin = np.mean((2.0 * y_true - 1.0) * (2.0 * prob - 1.0), axis=1)
    bce = -np.mean(y_true * np.log(prob) + (1 - y_true) * np.log(1.0 - prob), axis=1)
    rows = []
    for i, entry in enumerate(entries):
        tp = int(np.sum((y_true[i] == 1) & (y_pred[i] == 1)))
        fp = int(np.sum((y_true[i] == 0) & (y_pred[i] == 1)))
        fn = int(np.sum((y_true[i] == 1) & (y_pred[i] == 0)))
        denom = 2 * tp + fp + fn
        rows.append({'protein_id': str(entry), 'true_class_index': np.nan, 'true_class': None, 'predicted_class_index': np.nan, 'predicted_class': None, 'correct': int(np.array_equal(y_true[i], y_pred[i])), 'true_class_probability': np.nan, 'true_class_margin': np.nan, 'negative_log_likelihood': np.nan, 'protein_f1': float(2 * tp / denom) if denom else 1.0, 'signed_probability_margin': float(signed_margin[i]), 'binary_cross_entropy': float(bce[i])})
    return pd.DataFrame(rows)

def save_predictions(outdir, split, y_true, y_pred, y_prob, protein_rows):
    np.savez_compressed(outdir / 'predictions.npz', entries=np.asarray(split['test_entries'], dtype=str), label_names=np.asarray(split['labels'], dtype=str), y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    protein_rows.to_csv(outdir / 'predictions_per_protein.csv.gz', index=False, compression='gzip')

def run_one(args, task: str, emb_model: str, strategy: str, downstream: str) -> dict:
    start = time.time()
    outdir = Path(args.outdir) / emb_model / task / strategy / downstream
    done = outdir / 'metrics.json'
    prediction_file = outdir / 'predictions_per_protein.csv.gz'
    if done.exists() and prediction_file.exists() and (not args.overwrite):
        with done.open() as handle:
            metrics = json.load(handle)
        if metrics.get('validation_protocol') != 'train_only_tuning_frozen_val_thresholds_train_plus_val_refit':
            raise ValueError('Existing metrics use a different validation protocol; choose a new output directory.')
        metrics['skipped_existing'] = True
        return metrics
    split = slice_split(base.load_split(task), args.max_samples_per_split, args.random_state)
    kind = split['kind']
    path = embedding_path(emb_model, task, strategy)
    all_entries = np.concatenate([split['train_entries'], split['val_entries'], split['test_entries']])
    all_x = base.load_embeddings_in_order(path, all_entries)
    n_train, n_val = (len(split['train_entries']), len(split['val_entries']))
    x_train = all_x[:n_train]
    x_val = all_x[n_train:n_train + n_val]
    x_test = all_x[n_train + n_val:]
    y_train, y_val, y_test = (split['y_train'], split['y_val'], split['y_test'])
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'[START] task={task} emb={emb_model} strategy={strategy} downstream={downstream} kind={kind} X={x_train.shape}', flush=True)
    if downstream == 'lasso':
        model, best, thresholds = choose_lasso(kind, x_train, y_train, x_val, y_val, args.random_state)
        y_pred, y_prob = predict_model(kind, downstream, model, x_test, thresholds)
    elif downstream == 'xgboost':
        model, best, thresholds = choose_xgboost(kind, x_train, y_train, x_val, y_val, args.random_state, args.xgb_trials, args.num_workers)
        y_pred, y_prob = predict_model(kind, downstream, model, x_test, thresholds)
    elif downstream == 'dnn':
        model, scaler, best, thresholds = choose_dnn(kind, x_train, y_train, x_val, y_val, args.random_state, args)
        y_pred, y_prob = predict_model(kind, downstream, model, x_test, thresholds, scaler)
        torch.save(model.state_dict(), outdir / 'dnn_state_dict.pt')
    else:
        raise ValueError(downstream)
    if thresholds is not None:
        best['thresholds_selected_on_validation'] = thresholds.tolist()
    metrics = base.evaluate_multilabel(y_test, y_pred, y_prob)
    protein_rows = multilabel_protein_rows(split['test_entries'], y_test, y_pred, y_prob)
    protein_rows.insert(0, 'downstream_model', downstream)
    protein_rows.insert(0, 'strategy', strategy)
    protein_rows.insert(0, 'embedding_model', emb_model)
    protein_rows.insert(0, 'task', task)
    save_predictions(outdir, split, y_test, y_pred, y_prob, protein_rows)
    metrics.update({'source': 'true_truncation_independent_training', 'task': task, 'embedding_model': emb_model, 'strategy': strategy, 'downstream_model': downstream, 'task_kind': kind, 'embedding_path': str(path), 'train_n': int(len(x_train)), 'val_n': int(len(x_val)), 'test_n': int(len(x_test)), 'embedding_dim': int(x_train.shape[1]), 'label_n': int(len(split['labels'])), 'best_validation': best, 'runtime_seconds': round(time.time() - start, 3), 'smoke_test': bool(args.smoke_test), 'validation_protocol': 'train_only_tuning_frozen_val_thresholds_train_plus_val_refit'})
    base.save_json(done, metrics)
    print(f"[DONE] task={task} emb={emb_model} strategy={strategy} downstream={downstream} seconds={metrics['runtime_seconds']}", flush=True)
    return metrics

def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.max_samples_per_split = args.max_samples_per_split or 40
        args.xgb_trials = min(args.xgb_trials, 2)
        args.dnn_trials = min(args.dnn_trials, 2)
        args.dnn_max_epochs = min(args.dnn_max_epochs, 3)
        args.dnn_patience = min(args.dnn_patience, 2)
    if args.require_gpu and (not torch.cuda.is_available()):
        raise RuntimeError('--require-gpu was requested, but torch.cuda.is_available() is False')
    base.set_seed(args.random_state)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        for emb_model in args.embedding_models:
            for strategy in args.strategies:
                for downstream in args.downstream_models:
                    metrics = run_one(args, task, emb_model, strategy, downstream)
                    rows.append(metrics)
                    pd.DataFrame(rows).to_csv(outdir / 'summary_incremental.csv', index=False)
    pd.DataFrame(rows).to_csv(outdir / 'summary.csv', index=False)
    print(f"wrote {outdir / 'summary.csv'} rows={len(rows)}", flush=True)
if __name__ == '__main__':
    main()
