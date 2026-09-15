import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
MODELS = ['prott5', 'esm2']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Domain-pool saved full-length residue embeddings.')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--task', choices=['all'] + TASKS, default='all')
    parser.add_argument('--strategy-name', default='full_domain_pooling')
    return parser.parse_args()

def domain_mask(length: int, domains: list[list[int]]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start, end in domains:
        start_idx = max(0, int(start) - 1)
        end_idx = min(length, int(end))
        if end_idx > start_idx:
            mask[start_idx:end_idx] = True
    return mask

def summarize_domains(task: str, entries: list[str], lengths: dict[str, int], domain_map: dict[str, list[list[int]]]) -> list[dict]:
    rows = []
    for entry in entries:
        length = lengths[entry]
        domains = domain_map[entry]
        mask = domain_mask(length, domains)
        covered = int(mask.sum())
        rows.append({'task': task, 'entry': entry, 'length': length, 'domain_count': len(domains), 'domain_residues': covered, 'domain_fraction': covered / length if length else 0.0, 'first_domain_start': min((int(start) for start, _ in domains)), 'last_domain_end': max((int(end) for _, end in domains))})
    return rows

def main() -> None:
    args = parse_args()
    tasks = TASKS if args.task == 'all' else [args.task]
    residue_dir = ROOT / 'residue_embeddings' / args.model / 'all_unique'
    output_root = ROOT / 'embeddings_full_length' / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    domain_rows = []
    for task in tasks:
        master = pd.read_csv(ROOT / 'data' / f'{task}_master.csv', usecols=['Entry', 'Length'])
        entries = master['Entry'].astype(str).tolist()
        lengths = dict(zip(master['Entry'].astype(str), master['Length'].astype(int)))
        domain_map = json.loads((ROOT / 'data' / f'{task}_domains.json').read_text(encoding='utf-8'))
        payload = {}
        missing = []
        empty_domain = []
        domain_rows.extend(summarize_domains(task, entries, lengths, domain_map))
        for entry in entries:
            path = residue_dir / f'{entry}.npz'
            if not path.exists():
                missing.append(entry)
                continue
            residues = np.load(path)['residues']
            mask = domain_mask(residues.shape[0], domain_map[entry])
            if not mask.any():
                empty_domain.append(entry)
                payload[entry] = residues.astype(np.float32).mean(axis=0)
            else:
                payload[entry] = residues[mask].astype(np.float32).mean(axis=0)
        if missing:
            raise FileNotFoundError(f'{args.model} {task} missing {len(missing)} residue files')
        output = output_root / task / f'{args.strategy_name}.npz'
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **payload)
        first = next(iter(payload.values()))
        audit_rows.append({'model': args.model, 'task': task, 'strategy': args.strategy_name, 'entries': len(payload), 'embedding_dim': int(first.shape[-1]), 'empty_domain_fallbacks': len(empty_domain), 'output': str(output)})
        print(f'wrote {output} entries={len(payload)} dim={first.shape[-1]} empty_domain_fallbacks={len(empty_domain)}')
    stats_dir = ROOT / 'stats'
    stats_dir.mkdir(exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(stats_dir / f'{args.model}_{args.strategy_name}_pooling_audit.csv', index=False)
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(stats_dir / f'{args.model}_{args.strategy_name}_domain_distribution_entries.csv', index=False)
    summary = domain_df.groupby('task').agg(entries=('entry', 'count'), mean_length=('length', 'mean'), mean_domain_count=('domain_count', 'mean'), median_domain_count=('domain_count', 'median'), mean_domain_residues=('domain_residues', 'mean'), median_domain_residues=('domain_residues', 'median'), mean_domain_fraction=('domain_fraction', 'mean'), median_domain_fraction=('domain_fraction', 'median')).reset_index()
    summary.to_csv(stats_dir / f'{args.model}_{args.strategy_name}_domain_distribution_summary.csv', index=False)
if __name__ == '__main__':
    main()
