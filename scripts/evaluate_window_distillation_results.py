"""Independently train downstream models for distilled-window validation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import evaluate_downstream_strategies as base
import evaluate_true_truncation_results as true_eval
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
WINDOW_STRATEGIES = ['student_window', 'teacher_window', 'head500', *[f'random_seed{s}' for s in [101, 102, 103, 104, 105]]]
STRATEGIES = [*WINDOW_STRATEGIES, 'full_length']
DOWNSTREAM = ['lasso', 'xgboost', 'dnn']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=[s for s in STRATEGIES if s != 'head500'])
    parser.add_argument('--downstream-models', nargs='+', choices=DOWNSTREAM, default=DOWNSTREAM)
    parser.add_argument('--embedding-root', type=Path, default=ROOT / 'embeddings_window_distillation')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'downstream_window_distillation_results')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--xgb-trials', type=int, default=12)
    parser.add_argument('--dnn-trials', type=int, default=12)
    parser.add_argument('--dnn-max-epochs', type=int, default=50)
    parser.add_argument('--dnn-patience', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def embedding_path(args, task: str, strategy: str) -> Path:
    if strategy == 'full_length':
        return ROOT / 'embeddings_full_length' / 'esm2' / task / 'full_length_mean.npz'
    return args.embedding_root / 'esm2' / task / strategy / 'embeddings.npz'

def run_one(args, task: str, strategy: str, downstream: str) -> dict:
    started = time.time()
    out = args.outdir / 'esm2' / task / strategy / downstream
    metrics_path = out / 'metrics.json'
    protein_path = out / 'predictions_per_protein.csv.gz'
    if metrics_path.exists() and protein_path.exists() and (not args.overwrite):
        row = json.loads(metrics_path.read_text())
        row['skipped_existing'] = True
        return row
    split = base.load_split(task)
    entries = np.concatenate([split['train_entries'], split['val_entries'], split['test_entries']])
    path = embedding_path(args, task, strategy)
    x = base.load_embeddings_in_order(path, entries)
    n_train, n_val = (len(split['train_entries']), len(split['val_entries']))
    x_train, x_val, x_test = (x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:])
    y_train, y_val, y_test = (split['y_train'], split['y_val'], split['y_test'])
    kind = split['kind']
    print(f'[START] {task} esm2 {strategy} {downstream} X={x_train.shape}', flush=True)
    if downstream == 'lasso':
        fitted, best, thresholds = true_eval.choose_lasso(kind, x_train, y_train, x_val, y_val, args.random_state)
        y_pred, y_prob = true_eval.predict_model(kind, downstream, fitted, x_test, thresholds)
    elif downstream == 'xgboost':
        fitted, best, thresholds = true_eval.choose_xgboost(kind, x_train, y_train, x_val, y_val, args.random_state, args.xgb_trials, args.num_workers)
        y_pred, y_prob = true_eval.predict_model(kind, downstream, fitted, x_test, thresholds)
    else:
        fitted, scaler, best, thresholds = true_eval.choose_dnn(kind, x_train, y_train, x_val, y_val, args.random_state, args)
        y_pred, y_prob = true_eval.predict_model(kind, downstream, fitted, x_test, thresholds, scaler)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(fitted.state_dict(), out / 'dnn_state_dict.pt')
    if thresholds is not None:
        best['thresholds_selected_on_validation'] = thresholds.tolist()
    metrics = base.evaluate_multilabel(y_test, y_pred, y_prob)
    proteins = true_eval.multilabel_protein_rows(split['test_entries'], y_test, y_pred, y_prob)
    for column, value in reversed([('task', task), ('embedding_model', 'esm2'), ('strategy', strategy), ('downstream_model', downstream)]):
        proteins.insert(0, column, value)
    out.mkdir(parents=True, exist_ok=True)
    true_eval.save_predictions(out, split, y_test, y_pred, y_prob, proteins)
    metrics.update({'source': 'cross_fitted_teacher_student_window_distillation', 'task': task, 'embedding_model': 'esm2', 'strategy': strategy, 'downstream_model': downstream, 'task_kind': kind, 'embedding_path': str(path), 'train_n': len(x_train), 'val_n': len(x_val), 'test_n': len(x_test), 'embedding_dim': x.shape[1], 'best_validation': best, 'runtime_seconds': round(time.time() - started, 3), 'validation_protocol': 'train_only_tuning_frozen_val_thresholds_train_plus_val_refit'})
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f'[DONE] {task} esm2 {strategy} {downstream}', flush=True)
    return metrics

def main() -> None:
    args = parse_args()
    if args.require_gpu and (not torch.cuda.is_available()):
        raise RuntimeError('GPU is required but CUDA is unavailable')
    base.set_seed(args.random_state)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        for strategy in args.strategies:
            for downstream in args.downstream_models:
                rows.append(run_one(args, task, strategy, downstream))
                pd.DataFrame(rows).to_csv(args.outdir / 'summary_incremental.csv', index=False)
    pd.DataFrame(rows).to_csv(args.outdir / 'summary.csv', index=False)
    prediction_files = sorted(args.outdir.glob('*/*/*/*/predictions_per_protein.csv.gz'))
    predictions = pd.concat([pd.read_csv(path) for path in prediction_files], ignore_index=True)
    predictions.to_csv(args.outdir / 'predictions_per_protein.csv.gz', index=False, compression='gzip')
    print(f'[DONE] wrote {len(rows)} window-distillation evaluations', flush=True)
if __name__ == '__main__':
    main()
