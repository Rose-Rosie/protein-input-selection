"""Build cross-fitted teacher distributions for 500-aa windows.

For each task, L1 logistic teachers are fitted only on development proteins that
exclude the protein being scored.  A separate development-only teacher scores
the untouched test split.  Residue scores are converted to a continuous
distribution over every legal contiguous window; held-out labels are never
used while scoring residues or selecting windows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import residue_scoring as geometry
import build_signal_fragment_coordinates as fragments
import evaluate_downstream_strategies as base
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
RANDOM_SEEDS = [101, 102, 103, 104, 105]
SCHEMA_VERSION = 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--window-length', type=int, default=500)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-proteins-per-split', type=int, default=None)
    parser.add_argument('--outdir', type=Path, default=ROOT / 'analysis_outputs' / 'window_distillation' / 'teacher')
    return parser.parse_args()

def load_sequences() -> pd.DataFrame:
    frame = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Sequence', 'Length'], dtype={'Entry': str, 'Sequence': str}).set_index('Entry')
    if frame.index.duplicated().any():
        raise ValueError('authoritative sequence table contains duplicate Entry values')
    return frame

def tune_and_fit_coefficients(kind: str, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Select C internally and refit without touching an outer held-out protein."""
    if len(x) < 10:
        raise ValueError('teacher fit requires at least ten proteins')
    n_val = max(1, int(round(0.2 * len(x))))
    indices = np.arange(len(x))
    stratify = None
    train_idx, val_idx = train_test_split(indices, test_size=n_val, random_state=seed, shuffle=True, stratify=stratify)
    outer = StandardScaler().fit(x[train_idx])
    x_train = outer.transform(x[train_idx]).astype(np.float32)
    x_val = outer.transform(x[val_idx]).astype(np.float32)
    best = base.tune_lasso(kind, x_train, y[train_idx], x_val, y[val_idx], seed)
    final_outer = StandardScaler().fit(x)
    x_all = final_outer.transform(x).astype(np.float32)
    empty_x = np.empty((0, x_all.shape[1]), dtype=np.float32)
    fitted = base.fit_lasso(kind, x_all, y, empty_x, y[:0], best, seed)
    coeff = geometry.effective_coefficients(kind, fitted, final_outer, x.shape[1])
    audit = {'best_C': float(best['C']), 'internal_validation_score': float(best['score']), 'fit_proteins': int(len(x)), 'nonzero_coefficients': int(np.count_nonzero(coeff))}
    return (coeff.astype(np.float32, copy=False), audit)

def window_distribution(signal: np.ndarray, length: int, temperature: float) -> np.ndarray:
    masses = fragments.rolling_mass(signal, length).astype(np.float64)
    scale = float(masses.std())
    logits = (masses - float(masses.mean())) / max(scale, 1e-12) / temperature
    logits -= float(logits.max())
    probability = np.exp(logits)
    probability /= probability.sum()
    return probability.astype(np.float32)

def stable_random_start(task: str, entry: str, seed: int, max_start: int) -> int:
    token = f'window_distillation|{task}|{entry}|500|{seed}'.encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], 'little')
    return int(np.random.default_rng(derived).integers(0, max_start + 1))

def write_ragged(path: Path, rows: list[dict], probabilities: list[np.ndarray]) -> None:
    lengths = np.asarray([len(x) for x in probabilities], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    values = np.concatenate(probabilities).astype(np.float32, copy=False)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, entries=np.asarray([row['entry'] for row in rows], dtype=str), splits=np.asarray([row['split'] for row in rows], dtype=str), teacher_fold=np.asarray([row['teacher_fold'] for row in rows], dtype=int), lengths=lengths, offsets=offsets, probability=values, schema_version=np.asarray(SCHEMA_VERSION))
    tmp.replace(path)

def main() -> None:
    args = parse_args()
    if args.window_length <= 0 or args.folds < 2 or args.temperature <= 0:
        raise ValueError('window length, folds, and temperature must be positive')
    args.outdir.mkdir(parents=True, exist_ok=True)
    sequences = load_sequences()
    all_manifest: list[dict] = []
    all_folds: list[dict] = []
    for task in args.tasks:
        split = base.load_split(task)
        split_entries = {name: split[f'{name}_entries'].astype(str) for name in ['train', 'val', 'test']}
        split_labels = {name: split[f'y_{name}'] for name in ['train', 'val']}
        if args.max_proteins_per_split is not None:
            for name in split_entries:
                split_entries[name] = split_entries[name][:args.max_proteins_per_split]
                if name in split_labels:
                    split_labels[name] = split_labels[name][:args.max_proteins_per_split]
        development_entries = np.concatenate([split_entries['train'], split_entries['val']])
        development_y = np.concatenate([split_labels['train'], split_labels['val']], axis=0)
        x_dev = geometry.load_matrix(geometry.strategy_path('esm2', task, 'full_length_mean'), development_entries)
        fold_assignment: dict[str, int] = {}
        fold_coefficients: dict[int, np.ndarray] = {}
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
        split_iterator = splitter.split(development_entries)
        for fold, (fit_idx, held_idx) in enumerate(split_iterator):
            coeff, audit = tune_and_fit_coefficients(split['kind'], x_dev[fit_idx], development_y[fit_idx], args.random_state + fold)
            fold_coefficients[fold] = coeff
            held = development_entries[held_idx]
            fit_entries = set(development_entries[fit_idx])
            if fit_entries.intersection(held):
                raise RuntimeError(f'cross-fit overlap detected for {task} fold={fold}')
            fold_assignment.update({str(entry): fold for entry in held})
            all_folds.append({'task': task, 'teacher_fold': fold, 'role': 'development_oof', 'heldout_proteins': int(len(held)), 'fit_heldout_overlap': 0, **audit})
        test_coeff, test_audit = tune_and_fit_coefficients(split['kind'], x_dev, development_y, args.random_state + 1000)
        all_folds.append({'task': task, 'teacher_fold': -1, 'role': 'test_development_only', 'heldout_proteins': int(len(split_entries['test'])), 'fit_heldout_overlap': int(len(set(development_entries).intersection(split_entries['test']))), **test_audit})
        task_rows: list[dict] = []
        task_probabilities: list[np.ndarray] = []
        for split_name in ['train', 'val', 'test']:
            entries = split_entries[split_name]
            for idx, entry in enumerate(entries, start=1):
                sequence = str(sequences.at[entry, 'Sequence']).strip().upper()
                if len(sequence) < args.window_length:
                    raise ValueError(f'{task}/{entry} shorter than requested window')
                fold = fold_assignment[entry] if split_name != 'test' else -1
                coeff = fold_coefficients[fold] if fold >= 0 else test_coeff
                residue_path = ROOT / 'residue_embeddings' / 'esm2' / 'all_unique' / f'{entry}.npz'
                with np.load(residue_path) as data:
                    residues = np.asarray(data['residues'], dtype=np.float32)
                if len(residues) != len(sequence):
                    raise ValueError(f'sequence/residue length mismatch: {task}/{entry}')
                signal = geometry.task_global_signal(coeff, residues)
                probability = window_distribution(signal, args.window_length, args.temperature)
                teacher_start = int(np.argmax(probability))
                row = {'task': task, 'entry': entry, 'split': split_name, 'teacher_fold': fold, 'sequence_length': len(sequence), 'window_length': args.window_length, 'candidate_windows': len(probability), 'teacher_start_0based': teacher_start, 'teacher_entropy': float(-(probability * np.log(np.maximum(probability, 1e-30))).sum()), 'teacher_effective_windows': float(np.exp(-(probability * np.log(np.maximum(probability, 1e-30))).sum())), 'teacher_uses_entry_label': False, 'teacher_uses_test_labels': False, 'distribution_definition': 'softmax(zscore(rolling_500_sum(task_global_rms_projection)))'}
                for seed in RANDOM_SEEDS:
                    row[f'random_seed{seed}_start_0based'] = stable_random_start(task, entry, seed, len(sequence) - args.window_length)
                task_rows.append(row)
                task_probabilities.append(probability)
                if idx % 250 == 0 or idx == len(entries):
                    print(f'[TEACHER] task={task} split={split_name} {idx}/{len(entries)}', flush=True)
        task_frame = pd.DataFrame(task_rows)
        task_frame.to_csv(args.outdir / f'{task}__teacher_manifest.csv.gz', index=False)
        write_ragged(args.outdir / f'{task}__teacher_distribution.npz', task_rows, task_probabilities)
        all_manifest.extend(task_rows)
    pd.DataFrame(all_manifest).to_csv(args.outdir / 'teacher_manifest.csv.gz', index=False)
    pd.DataFrame(all_folds).to_csv(args.outdir / 'teacher_fold_audit.csv', index=False)
    audit = {'schema_version': SCHEMA_VERSION, 'embedding_model': 'esm2', 'window_length': args.window_length, 'crossfit_folds': args.folds, 'teacher_distribution_temperature': args.temperature, 'heldout_entry_labels_used_for_teacher_distribution': False, 'test_labels_used_anywhere': False, 'rows': len(all_manifest), 'tasks': args.tasks}
    (args.outdir / 'teacher_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'[DONE] cross-fitted teacher distributions rows={len(all_manifest)}', flush=True)
if __name__ == '__main__':
    main()
