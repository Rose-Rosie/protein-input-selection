import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
MODELS = ['prott5', 'esm2']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Mean-pool saved full-length residue embeddings.')
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--task', choices=['all'] + TASKS, default='all')
    parser.add_argument('--strategy-name', default='full_length_mean')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    tasks = TASKS if args.task == 'all' else [args.task]
    residue_dir = ROOT / 'residue_embeddings' / args.model / 'all_unique'
    output_root = ROOT / 'embeddings_full_length' / args.model
    output_root.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for task in tasks:
        entries = pd.read_csv(ROOT / 'data' / f'{task}_master.csv', usecols=['Entry'])['Entry'].astype(str).tolist()
        payload = {}
        missing = []
        for entry in entries:
            path = residue_dir / f'{entry}.npz'
            if not path.exists():
                missing.append(entry)
                continue
            residues = np.load(path)['residues']
            payload[entry] = residues.astype(np.float32).mean(axis=0)
        if missing:
            raise FileNotFoundError(f'{args.model} {task} missing {len(missing)} residue files')
        output = output_root / task / f'{args.strategy_name}.npz'
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **payload)
        first = next(iter(payload.values()))
        audit_rows.append({'model': args.model, 'task': task, 'strategy': args.strategy_name, 'entries': len(payload), 'embedding_dim': int(first.shape[-1]), 'output': str(output)})
        print(f'wrote {output} entries={len(payload)} dim={first.shape[-1]}')
    stats_dir = ROOT / 'stats'
    stats_dir.mkdir(exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(stats_dir / f'{args.model}_{args.strategy_name}_pooling_audit.csv', index=False)
if __name__ == '__main__':
    main()
