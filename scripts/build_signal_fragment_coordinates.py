from __future__ import annotations
from project_paths import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import residue_scoring as geometry
import evaluate_downstream_strategies as base

def fit_train_only_selector(task: str, model: str, seed: int, retune: bool) -> dict:
    """Tune on train/validation, then fit selector coefficients on train only."""
    split = base.load_split(task)
    entries = np.concatenate([split['train_entries'], split['val_entries'], split['test_entries']]).astype(str)
    x_all = geometry.load_matrix(geometry.strategy_path(model, task, 'full_length_mean'), entries)
    n_train = len(split['train_entries'])
    n_val = len(split['val_entries'])
    x_train = x_all[:n_train]
    x_val = x_all[n_train:n_train + n_val]
    outer = StandardScaler().fit(x_train)
    x_train_s = outer.transform(x_train).astype(np.float32)
    x_val_s = outer.transform(x_val).astype(np.float32)
    best = None
    if best is None:
        best = base.tune_lasso(split['kind'], x_train_s, split['y_train'], x_val_s, split['y_val'], seed)
        best['source'] = 'retuned_by_fragment_builder'
    empty_x = np.empty((0, x_train_s.shape[1]), dtype=np.float32)
    train_only_model = base.fit_lasso(split['kind'], x_train_s, split['y_train'], empty_x, split['y_train'][:0], best, seed)
    coeff = geometry.effective_coefficients(split['kind'], train_only_model, outer, dim=x_train.shape[1])
    return {'split': split, 'entries': entries, 'coeff': coeff.astype(np.float32), 'best': best, 'embedding_dim': int(x_train.shape[1])}

def contribution_matrix(residues: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return residues.astype(np.float32, copy=False) @ coeff.T.astype(np.float32, copy=False)

def rms_signal(contributions: np.ndarray, indices: np.ndarray | None=None) -> np.ndarray:
    selected = contributions if indices is None else contributions[:, indices]
    if selected.ndim == 1:
        selected = selected[:, None]
    if selected.shape[1] == 0:
        return np.zeros(selected.shape[0], dtype=np.float32)
    return np.sqrt(np.mean(np.square(selected, dtype=np.float64), axis=1)).astype(np.float32)

def rolling_mass(signal: np.ndarray, length: int) -> np.ndarray:
    if length > len(signal):
        raise ValueError(f'fragment length {length} exceeds sequence length {len(signal)}')
    prefix = np.concatenate(([0.0], np.cumsum(signal, dtype=np.float64)))
    return prefix[length:] - prefix[:-length]

def stable_random_start(task: str, model: str, entry: str, mode: str, length: int, seed: int, max_start: int) -> int:
    token = f'{task}|{model}|{entry}|{mode}|{length}|{seed}'.encode('utf-8')
    digest = hashlib.sha256(token).digest()
    derived_seed = int.from_bytes(digest[:8], byteorder='little', signed=False)
    rng = np.random.default_rng(derived_seed)
    return int(rng.integers(0, max_start + 1))

def fragment_hash(sequence: str, start: int, length: int) -> str:
    return hashlib.sha256(sequence[start:start + length].encode('ascii')).hexdigest()

def main():
    import argparse
    from project_paths import TASKS
    p=argparse.ArgumentParser(description='Select 300-residue windows using training-only linear scores.')
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    p.add_argument('--embedding-models',nargs='+',choices=['esm2','prott5'],default=['esm2','prott5'])
    p.add_argument('--random-state',type=int,default=42)
    args=p.parse_args()
    sequences=pd.read_csv(ROOT/'data/all_unique_full_length_entries.csv').set_index('Entry')['Sequence']
    rows=[]
    for task in args.tasks:
        for model in args.embedding_models:
            fitted=fit_train_only_selector(task,model,args.random_state,True)
            for split_name in ['train','val','test']:
                for entry in fitted['split'][f'{split_name}_entries']:
                    sequence=str(sequences.at[entry])
                    with np.load(ROOT/'residue_embeddings'/model/'all_unique'/f'{entry}.npz') as data:
                        residues=data['residues']
                    if len(residues)!=len(sequence):raise ValueError(f'Residue count mismatch: {entry}')
                    signal=rms_signal(contribution_matrix(residues,fitted['coeff']))
                    masses=rolling_mass(signal,300)
                    starts=[('high_signal',None,int(np.argmax(masses)))]
                    starts += [('random',seed,stable_random_start(task,model,entry,'task_global',300,seed,len(sequence)-300)) for seed in [101,102,103,104,105]]
                    for kind,seed,start in starts:
                        rows.append(dict(task=task,embedding_model=model,split=split_name,entry=entry,
                          selection_mode='task_global',selection_uses_ground_truth=False,
                          fragment_length=300,fragment_type=kind,random_seed=seed,
                          start_0based=start,end_0based_exclusive=start+300,
                          fragment_sha256=fragment_hash(sequence,start,300)))
    output=ROOT/'analysis_outputs/signal_fragments/fragment_coordinate_manifest.csv.gz'
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False)
    print(f'Wrote {len(rows)} windows to {output}')

if __name__=='__main__':main()
