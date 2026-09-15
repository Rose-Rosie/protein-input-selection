"""Encode only student/teacher/random 500-aa windows with ESM2."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import encode_signal_fragments as signal_encoder
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
STRATEGIES = ['student_window', 'teacher_window', 'head500', *[f'random_seed{s}' for s in [101, 102, 103, 104, 105]]]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=[s for s in STRATEGIES if s != 'head500'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--shard-size', type=int, default=250)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'analysis_outputs' / 'window_distillation' / 'student' / 'selected_window_manifest.csv.gz')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'embeddings_window_distillation')
    parser.add_argument('--audit-path', type=Path, default=ROOT / 'analysis_outputs' / 'window_distillation' / 'embedding_audit.csv')
    return parser.parse_args()

def load_sequences() -> dict[str, str]:
    frame = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Sequence'], dtype={'Entry': str, 'Sequence': str})
    return dict(zip(frame.Entry, frame.Sequence))

def load_rows(args, task: str, strategy: str) -> pd.DataFrame:
    frame = pd.read_csv(args.manifest, dtype={'entry': str})
    frame = frame[(frame.task == task) & (frame.strategy == strategy)].copy()
    frame['split_order'] = frame['split'].map({'train': 0, 'val': 1, 'test': 2})
    frame = frame.sort_values(['split_order', 'entry']).reset_index(drop=True)
    if args.limit is not None:
        frame = frame.groupby('split', group_keys=False).head(args.limit).reset_index(drop=True)
    return frame

def validate_fragment(row, sequence: str) -> str:
    start, end = (int(row.start_0based), int(row.end_0based_exclusive))
    fragment = sequence[start:end]
    expected = hashlib.sha256(fragment.encode()).hexdigest()
    if len(fragment) != int(row.window_length) or expected != row.fragment_sha256:
        raise ValueError(f'fragment validation failed for {row.task}/{row.entry}/{row.strategy}')
    return fragment

def run_strategy(args, task, strategy, rows, sequences, bundle, device) -> None:
    out = args.output_root / 'esm2' / task / strategy
    for shard_start in range(0, len(rows), args.shard_size):
        shard = rows.iloc[shard_start:shard_start + args.shard_size]
        part = out / 'parts' / f'part_{shard_start:07d}.npz'
        if part.exists() and (not args.overwrite):
            continue
        started = time.perf_counter()
        entries, vectors = ([], [])
        batch_entries, batch_sequences = ([], [])
        for row in shard.itertuples(index=False):
            batch_entries.append(str(row.entry))
            batch_sequences.append(validate_fragment(row, sequences[str(row.entry)]))
            if len(batch_entries) == args.batch_size:
                vectors.extend(signal_encoder.embed_batch('esm2', bundle, device, batch_entries, batch_sequences))
                entries.extend(batch_entries)
                batch_entries, batch_sequences = ([], [])
        if batch_entries:
            vectors.extend(signal_encoder.embed_batch('esm2', bundle, device, batch_entries, batch_sequences))
            entries.extend(batch_entries)
        elapsed = time.perf_counter() - started
        signal_encoder.write_shard(part, entries, vectors, {'task': task, 'embedding_model': 'esm2', 'strategy': strategy, 'rows': len(entries), 'window_length': 500, 'encoded_residues': int(500 * len(entries)), 'elapsed_seconds': elapsed, 'seconds_per_protein': elapsed / max(len(entries), 1)})
        print(f'[SHARD] {task} {strategy} start={shard_start} rows={len(entries)} seconds={elapsed:.3f}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    payload: dict[str, np.ndarray] = {}
    for part in sorted((out / 'parts').glob('part_*.npz')):
        with np.load(part, allow_pickle=True) as data:
            for key in data.files:
                if key == '__metadata_json__':
                    continue
                if key in payload:
                    raise ValueError(f'duplicate entry while merging {part}: {key}')
                payload[key] = np.asarray(data[key], dtype=np.float32)
    expected = set(rows.entry.astype(str))
    if set(payload) != expected:
        raise ValueError(f'merge inventory mismatch {task}/{strategy}: expected={len(expected)} got={len(payload)}')
    out.mkdir(parents=True, exist_ok=True)
    final = out / 'embeddings.npz'
    tmp = final.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(final)
    print(f'[MERGED] {final} entries={len(payload)}', flush=True)

def collect_audit(args) -> None:
    rows = []
    for part in sorted(args.output_root.glob('esm2/*/*/parts/part_*.npz')):
        with np.load(part, allow_pickle=True) as data:
            metadata = json.loads(str(data['__metadata_json__'].item()))
        metadata['part_path'] = str(part)
        rows.append(metadata)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.audit_path, index=False)

def main() -> None:
    args = parse_args()
    sequences = load_sequences()
    bundle, device = signal_encoder.load_model('esm2')
    print(f'[MODEL] esm2 device={device} batch_size={args.batch_size}', flush=True)
    for task in args.tasks:
        for strategy in args.strategies:
            rows = load_rows(args, task, strategy)
            run_strategy(args, task, strategy, rows, sequences, bundle, device)
    collect_audit(args)
    print('[DONE] ESM2 window-distillation encoding', flush=True)
if __name__ == '__main__':
    main()
