from project_paths import ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def interval_mask(length: int, intervals: list[list[int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        start_idx = max(0, int(start) - 1)
        end_idx = min(length, int(end))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask

def contiguous_runs(mask: np.ndarray) -> list[int]:
    lengths = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            lengths.append(i - start)
            start = None
    if start is not None:
        lengths.append(len(mask) - start)
    return lengths

def random_matched_mask(length: int, run_lengths: list[int], rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if length == 0:
        return mask
    for run_len in sorted([n for n in run_lengths if n > 0], reverse=True):
        run_len = min(run_len, length)
        candidates = []
        for start in range(0, length - run_len + 1):
            if not mask[start:start + run_len].any():
                candidates.append(start)
        if candidates:
            start = int(rng.choice(candidates))
            mask[start:start + run_len] = True
            continue
        available = np.flatnonzero(~mask)
        if len(available) == 0:
            break
        chosen = rng.choice(available, size=min(run_len, len(available)), replace=False)
        mask[chosen] = True
    return mask

def pooled_vector(residues: np.ndarray, mask: np.ndarray, empty_policy: str) -> np.ndarray:
    if mask.any():
        return residues[mask].astype(np.float32).mean(axis=0)
    if empty_policy == 'full_mean':
        return residues.astype(np.float32).mean(axis=0)
    return np.zeros(residues.shape[1], dtype=np.float32)

def main():
    import argparse
    from project_paths import TASKS
    p=argparse.ArgumentParser()
    p.add_argument('--models',nargs='+',choices=['esm2','prott5'],default=['esm2','prott5'])
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    args=p.parse_args()
    for model in args.models:
        for task in args.tasks:
            entries=pd.read_csv(ROOT/'data'/f'{task}_master.csv')['Entry'].astype(str)
            domains=json.loads((ROOT/'data'/f'{task}_domains.json').read_text())
            payloads={seed:{} for seed in [101,102,103,104,105]}
            for entry in entries:
                with np.load(ROOT/'residue_embeddings'/model/'all_unique'/f'{entry}.npz') as z:residues=z['residues']
                domain=interval_mask(len(residues),domains[entry]); runs=contiguous_runs(domain)
                for seed,payload in payloads.items():
                    rng=np.random.default_rng(sum(ord(c) for c in entry)+seed*1000003)
                    mask=random_matched_mask(len(residues),runs,rng)
                    if mask.sum()!=domain.sum():raise ValueError('Random control residue-count mismatch')
                    payload[entry]=pooled_vector(residues,mask,'zero')
            out=ROOT/'embeddings_region_ablation'/model/task;out.mkdir(parents=True,exist_ok=True)
            for seed,payload in payloads.items():np.savez(out/f'domain_matched_random_region_only_seed{seed}.npz',**payload)
if __name__=='__main__':main()
