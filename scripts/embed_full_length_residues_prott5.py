import argparse
import csv
import gc
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Save full-length ProtT5 residue embeddings.')
    parser.add_argument('--task', choices=['all'] + TASKS, default='all')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'residue_embeddings' / 'prott5')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def clean_seq(seq: str) -> str:
    return re.sub('[UZOB]', 'X', str(seq).upper().strip())

def to_spaced(seq: str) -> str:
    return ' '.join(list(seq))

def load_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(model_name)
    if device.type == 'cpu':
        model.to(torch.float32)
    model = model.to(device).eval()
    return (tokenizer, model, device)

def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def embed_one(tokenizer, model, device, sequence: str) -> np.ndarray:
    cleaned = clean_seq(sequence)
    spaced = to_spaced(cleaned)
    ids = tokenizer([spaced], return_tensors='pt', padding=True, add_special_tokens=True).to(device)
    with torch.inference_mode():
        hidden = model(**ids).last_hidden_state
    mask = ids.attention_mask[0].bool()
    valid = hidden[0][mask]
    if valid.shape[0] == len(cleaned) + 1:
        residues = valid[:-1]
    elif valid.shape[0] == len(cleaned) + 2:
        residues = valid[1:-1]
    elif valid.shape[0] == len(cleaned):
        residues = valid
    else:
        raise ValueError(f'Unexpected ProtT5 token count: valid={valid.shape[0]} length={len(cleaned)}')
    residues = residues.detach().cpu().numpy()
    if residues.shape[0] != len(cleaned):
        raise ValueError(f'Residue shape mismatch: residues={residues.shape[0]} length={len(cleaned)}')
    return residues

def run_task(task: str, args: argparse.Namespace, tokenizer, model, device) -> None:
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
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f'ProtT5 {task}'):
        entry = str(row.Entry)
        output = out_dir / f'{entry}.npz'
        if output.exists() and (not args.overwrite):
            continue
        started = time.perf_counter()
        status = 'ok'
        error = ''
        shape = ''
        try:
            residues = embed_one(tokenizer, model, device, row.Sequence)
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
    tokenizer, model, device = load_model()
    tasks = ['all'] if args.task == 'all' else [args.task]
    for task in tasks:
        run_task(task, args, tokenizer, model, device)
if __name__ == '__main__':
    main()
