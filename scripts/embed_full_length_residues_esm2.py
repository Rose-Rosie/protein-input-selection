import argparse
import csv
import gc
import re
import time
from pathlib import Path
import esm
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save full-length ESM2 residue embeddings.')
    parser.add_argument('--task', choices=['all'] + TASKS, default='all')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'residue_embeddings' / 'esm2')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def clean_seq(seq: str) -> str:
    return re.sub('[UZOB]', 'X', str(seq).upper().strip())

def load_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return (alphabet.get_batch_converter(), model, device)

def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def embed_one(batch_converter, model, device, entry: str, sequence: str) -> np.ndarray:
    cleaned = clean_seq(sequence)
    _, _, tokens = batch_converter([(entry, cleaned)])
    tokens = tokens.to(device)
    with torch.inference_mode():
        output = model(tokens, repr_layers=[33], return_contacts=False)
        hidden = output['representations'][33]
    padding_idx = 1
    batch_len = int((tokens != padding_idx).sum(1)[0].item())
    residues = hidden[0, 1:batch_len - 1].detach().cpu().numpy()
    if residues.shape[0] != len(cleaned):
        raise ValueError(f'Residue shape mismatch: residues={residues.shape[0]} length={len(cleaned)}')
    return residues

def run_task(task: str, args: argparse.Namespace, batch_converter, model, device) -> None:
    if task == 'all':
        df = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Sequence', 'Length'])
        task_name = 'all_unique'
    else:
        df = pd.read_csv(ROOT / 'data' / f'{task}_master.csv', usecols=['Entry', 'Sequence', 'Length'])
        task_name = task
    if args.limit is not None:
        df = df.head(args.limit).copy()
    out_dir = args.output_root / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / f'{task_name}_manifest.csv'
    dtype = np.float16 if args.dtype == 'float16' else np.float32
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f'ESM2 {task}'):
        entry = str(row.Entry)
        output = out_dir / f'{entry}.npz'
        if output.exists() and (not args.overwrite):
            continue
        started = time.perf_counter()
        status = 'ok'
        error = ''
        shape = ''
        try:
            residues = embed_one(batch_converter, model, device, entry, row.Sequence)
            shape = 'x'.join(map(str, residues.shape))
            np.savez_compressed(output, residues=residues.astype(dtype), entry=entry, length=int(row.Length))
        except Exception as exc:
            status = 'error'
            error = ' '.join(str(exc).split())[:500]
        elapsed = time.perf_counter() - started
        append_manifest(manifest, {'task': task, 'entry': entry, 'length': int(row.Length), 'status': status, 'shape': shape, 'output': str(output), 'elapsed_seconds': round(elapsed, 4), 'error': error})
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

def main() -> None:
    args = parse_args()
    batch_converter, model, device = load_model()
    tasks = ['all'] if args.task == 'all' else [args.task]
    for task in tasks:
        run_task(task, args, batch_converter, model, device)
if __name__ == '__main__':
    main()
