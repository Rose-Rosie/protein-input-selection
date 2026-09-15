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
MODELS = ['prott5', 'esm2']
STRATEGIES = ['high_signal', 'random_seed101', 'random_seed102', 'random_seed103', 'random_seed104', 'random_seed105']
DOWNSTREAM = ['lasso', 'xgboost', 'dnn']

def parse_args():
    p = argparse.ArgumentParser(description='Independently train downstream models for signal-fragment representations.')
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--embedding-models', nargs='+', choices=MODELS, default=MODELS)
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    p.add_argument('--downstream-models', nargs='+', choices=DOWNSTREAM, default=DOWNSTREAM)
    p.add_argument('--selection-mode', choices=['task_global'], default='task_global')
    p.add_argument('--fragment-length', type=int, choices=[300], default=300)
    p.add_argument('--embedding-root', type=Path, default=ROOT / 'embeddings_signal_fragments')
    p.add_argument('--outdir', type=Path, default=ROOT / 'downstream_signal_fragment_results')
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--xgb-trials', type=int, default=12)
    p.add_argument('--dnn-trials', type=int, default=12)
    p.add_argument('--dnn-max-epochs', type=int, default=50)
    p.add_argument('--dnn-patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def embedding_path(args, model, task, strategy):
    return args.embedding_root / model / task / args.selection_mode / str(args.fragment_length) / strategy / 'embeddings.npz'

def run_one(args, task, model_name, strategy, downstream):
    started = time.time()
    out = args.outdir / args.selection_mode / str(args.fragment_length) / model_name / task / strategy / downstream
    metrics_path = out / 'metrics.json'
    prediction_path = out / 'predictions_per_protein.csv.gz'
    if metrics_path.exists() and prediction_path.exists() and (not args.overwrite):
        with metrics_path.open() as handle:
            row = json.load(handle)
        row['skipped_existing'] = True
        return row
    split = base.load_split(task)
    kind = split['kind']
    entries = np.concatenate([split['train_entries'], split['val_entries'], split['test_entries']])
    path = embedding_path(args, model_name, task, strategy)
    x = base.load_embeddings_in_order(path, entries)
    n_train, n_val = (len(split['train_entries']), len(split['val_entries']))
    x_train, x_val, x_test = (x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:])
    y_train, y_val, y_test = (split['y_train'], split['y_val'], split['y_test'])
    print(f'[START] {task} {model_name} {strategy} {downstream} X={x_train.shape}', flush=True)
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
    for col, value in reversed([('task', task), ('embedding_model', model_name), ('selection_mode', args.selection_mode), ('fragment_length', args.fragment_length), ('strategy', strategy), ('downstream_model', downstream)]):
        proteins.insert(0, col, value)
    out.mkdir(parents=True, exist_ok=True)
    true_eval.save_predictions(out, split, y_test, y_pred, y_prob, proteins)
    metrics.update({'source': 'independently_encoded_signal_fragment', 'task': task, 'embedding_model': model_name, 'selection_mode': args.selection_mode, 'fragment_length': args.fragment_length, 'strategy': strategy, 'downstream_model': downstream, 'task_kind': kind, 'embedding_path': str(path), 'train_n': len(x_train), 'val_n': len(x_val), 'test_n': len(x_test), 'embedding_dim': x.shape[1], 'label_n': len(split['labels']), 'best_validation': best, 'runtime_seconds': round(time.time() - started, 3), 'validation_protocol': 'train_only_tuning_frozen_val_thresholds_train_plus_val_refit'})
    with metrics_path.open('w') as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[DONE] {task} {model_name} {strategy} {downstream} seconds={metrics['runtime_seconds']}", flush=True)
    return metrics

def combine_predictions(args):
    root = args.outdir / args.selection_mode / str(args.fragment_length)
    files = sorted(root.glob('*/*/*/*/predictions_per_protein.csv.gz'))
    if not files:
        return
    predictions = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    predictions.to_csv(root / 'predictions_per_protein.csv.gz', index=False, compression='gzip')

def main():
    args = parse_args()
    if args.require_gpu and (not torch.cuda.is_available()):
        raise RuntimeError('GPU is required but CUDA is unavailable')
    base.set_seed(args.random_state)
    rows = []
    summary_dir = args.outdir / args.selection_mode / str(args.fragment_length)
    summary_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for model_name in args.embedding_models:
            for strategy in args.strategies:
                for downstream in args.downstream_models:
                    rows.append(run_one(args, task, model_name, strategy, downstream))
                    pd.DataFrame(rows).to_csv(summary_dir / 'summary_incremental.csv', index=False)
    pd.DataFrame(rows).to_csv(summary_dir / 'summary.csv', index=False)
    combine_predictions(args)
    print(f'[DONE] wrote {len(rows)} signal-fragment evaluations', flush=True)
if __name__ == '__main__':
    main()
