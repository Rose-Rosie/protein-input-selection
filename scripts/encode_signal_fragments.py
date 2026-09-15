import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
SEEDS = [101, 102, 103, 104, 105]
STRATEGIES = ['high_signal', *[f'random_seed{s}' for s in SEEDS]]

def parse_args():
    p = argparse.ArgumentParser(description='Independently encode selected signal fragments and mean-pool them.')
    p.add_argument('--embedding-model', choices=['prott5', 'esm2'], required=True)
    p.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    p.add_argument('--selection-mode', choices=['task_global'], default='task_global')
    p.add_argument('--fragment-length', type=int, choices=[300], default=300)
    p.add_argument('--strategies', nargs='+', choices=STRATEGIES, default=STRATEGIES)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--shard-size', type=int, default=250)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--output-root', type=Path, default=ROOT / 'embeddings_signal_fragments')
    return p.parse_args()

def clean_seq(seq):
    return re.sub('[UZOB]', 'X', str(seq).upper().strip())

def load_sequences():
    df = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Sequence'])
    if df['Entry'].duplicated().any():
        raise ValueError('Authoritative sequence table contains duplicate Entry values')
    return dict(zip(df['Entry'].astype(str), df['Sequence'].astype(str)))

def load_rows(args, task):
    use = ['task', 'embedding_model', 'split', 'entry', 'selection_mode', 'fragment_length', 'fragment_type', 'random_seed', 'start_0based', 'end_0based_exclusive', 'fragment_sha256']
    df = pd.read_csv(ROOT / 'analysis_outputs' / 'signal_fragments' / 'fragment_coordinate_manifest.csv.gz', usecols=use)
    df = df[(df.task == task) & (df.embedding_model == args.embedding_model) & (df.selection_mode == args.selection_mode) & (df.fragment_length == args.fragment_length)].copy()
    df['strategy'] = df['fragment_type']
    random = df['fragment_type'].eq('random')
    df.loc[random, 'strategy'] = 'random_seed' + df.loc[random, 'random_seed'].astype(int).astype(str)
    expected = {'high_signal', *[f'random_seed{s}' for s in SEEDS]}
    if set(df.strategy) != expected:
        raise ValueError(f'Unexpected strategy inventory for {task}: {sorted(df.strategy.unique())}')
    if df.duplicated(['entry', 'strategy']).any():
        raise ValueError(f'Duplicate entry/strategy rows for {task}')
    split_order = {'train': 0, 'val': 1, 'test': 2}
    df['split_order'] = df['split'].map(split_order)
    df = df.sort_values(['strategy', 'split_order', 'entry']).reset_index(drop=True)
    if args.limit is not None:
        df = df.groupby('strategy', group_keys=False).head(args.limit).reset_index(drop=True)
    return df

def load_model(name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('GPU is required for formal fragment encoding')
    if name == 'prott5':
        from transformers import T5EncoderModel, T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False, legacy=True)
        model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device).eval()
        return ((tokenizer, model), device)
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    return ((alphabet.get_batch_converter(), model.to(device).eval()), device)

def embed_batch(name, bundle, device, entries, sequences):
    cleaned = [clean_seq(s) for s in sequences]
    with torch.inference_mode():
        if name == 'prott5':
            tokenizer, model = bundle
            spaced = [' '.join(list(s)) for s in cleaned]
            tok = tokenizer(spaced, return_tensors='pt', padding=True, add_special_tokens=True).to(device)
            hidden = model(**tok).last_hidden_state
            pooled = []
            for i, seq in enumerate(cleaned):
                valid = hidden[i][tok.attention_mask[i].bool()]
                if valid.shape[0] == len(seq) + 1:
                    residues = valid[:-1]
                elif valid.shape[0] == len(seq) + 2:
                    residues = valid[1:-1]
                elif valid.shape[0] == len(seq):
                    residues = valid
                else:
                    raise ValueError(f'ProtT5 token mismatch entry={entries[i]} tokens={valid.shape[0]} residues={len(seq)}')
                pooled.append(residues.float().mean(0).cpu().numpy())
        else:
            batch_converter, model = bundle
            _, _, tokens = batch_converter(list(zip(entries, cleaned)))
            tokens = tokens.to(device)
            hidden = model(tokens, repr_layers=[33], return_contacts=False)['representations'][33]
            pooled = []
            for i, seq in enumerate(cleaned):
                residues = hidden[i, 1:len(seq) + 1]
                if residues.shape[0] != len(seq):
                    raise ValueError(f'ESM2 token mismatch entry={entries[i]}')
                pooled.append(residues.float().mean(0).cpu().numpy())
    return pooled

def write_shard(path, entries, vectors, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {entry: np.asarray(vec, dtype=np.float32) for entry, vec in zip(entries, vectors)}
    payload['__metadata_json__'] = np.asarray(json.dumps(metadata, sort_keys=True))
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)

def validate_fragment(row, sequence):
    start, end = (int(row.start_0based), int(row.end_0based_exclusive))
    fragment = sequence[start:end]
    if len(fragment) != int(row.fragment_length):
        raise ValueError(f'Fragment length mismatch entry={row.entry}: {len(fragment)} != {row.fragment_length}')
    digest = hashlib.sha256(fragment.encode()).hexdigest()
    if digest != row.fragment_sha256:
        raise ValueError(f'Fragment SHA mismatch entry={row.entry} strategy={row.strategy}')
    return fragment

def run_strategy(args, task, strategy, rows, sequences, bundle, device):
    out = args.output_root / args.embedding_model / task / args.selection_mode / str(args.fragment_length) / strategy
    rows = rows[rows.strategy == strategy].reset_index(drop=True)
    for shard_start in range(0, len(rows), args.shard_size):
        shard = rows.iloc[shard_start:shard_start + args.shard_size]
        part = out / 'parts' / f'part_{shard_start:07d}.npz'
        if part.exists() and (not args.overwrite):
            continue
        entries, vectors = ([], [])
        batch_entries, batch_sequences = ([], [])
        for row in shard.itertuples(index=False):
            seq = sequences[str(row.entry)]
            fragment = validate_fragment(row, seq)
            batch_entries.append(str(row.entry))
            batch_sequences.append(fragment)
            if len(batch_entries) == args.batch_size:
                vectors.extend(embed_batch(args.embedding_model, bundle, device, batch_entries, batch_sequences))
                entries.extend(batch_entries)
                batch_entries, batch_sequences = ([], [])
        if batch_entries:
            vectors.extend(embed_batch(args.embedding_model, bundle, device, batch_entries, batch_sequences))
            entries.extend(batch_entries)
        write_shard(part, entries, vectors, {'task': task, 'embedding_model': args.embedding_model, 'selection_mode': args.selection_mode, 'fragment_length': args.fragment_length, 'strategy': strategy, 'rows': len(entries)})
        print(f'[SHARD] {task} {args.embedding_model} {strategy} start={shard_start} rows={len(entries)}', flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    final = out / 'embeddings.npz'
    if final.exists() and (not args.overwrite):
        return
    payload = {}
    for part in sorted((out / 'parts').glob('part_*.npz')):
        with np.load(part, allow_pickle=True) as data:
            for key in data.files:
                if key == '__metadata_json__':
                    continue
                if key in payload:
                    raise ValueError(f'Duplicate entry while merging {part}: {key}')
                payload[key] = np.asarray(data[key], dtype=np.float32)
    expected_entries = set(rows.entry.astype(str))
    if set(payload) != expected_entries:
        raise ValueError(f'Merge inventory mismatch {task}/{strategy}: expected={len(expected_entries)} got={len(payload)}')
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload)
    tmp.replace(final)
    print(f'[MERGED] {final} entries={len(payload)}', flush=True)

def main():
    args = parse_args()
    sequences = load_sequences()
    bundle, device = load_model(args.embedding_model)
    print(f'[MODEL] {args.embedding_model} device={device} batch_size={args.batch_size}', flush=True)
    for task in args.tasks:
        rows = load_rows(args, task)
        missing = sorted(set(rows.entry.astype(str)) - set(sequences))
        if missing:
            raise KeyError(f'Missing authoritative sequences for {task}: {missing[:5]}')
        for strategy in args.strategies:
            run_strategy(args, task, strategy, rows, sequences, bundle, device)
    print('[DONE] signal fragment encoding', flush=True)
if __name__ == '__main__':
    main()
