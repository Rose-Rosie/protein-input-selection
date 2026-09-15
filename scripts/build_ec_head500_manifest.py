"""Build the fixed-position EC_level2 N-terminal 500-aa control manifest."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from project_paths import ROOT
OUT = ROOT / 'analysis_outputs' / 'ec_demo' / 'head500_manifest.csv.gz'

def main() -> None:
    sequences = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Length', 'Sequence'], dtype={'Entry': str, 'Sequence': str}).set_index('Entry')
    split = np.load(ROOT / 'splits' / 'EC_level2_split_entries.npz', allow_pickle=True)
    rows = []
    for split_name in ['train', 'val', 'test']:
        entries = split[f'{split_name}_entries'].astype(str)
        for entry in entries:
            sequence = str(sequences.at[entry, 'Sequence']).strip().upper()
            declared = int(sequences.at[entry, 'Length'])
            if len(sequence) != declared or len(sequence) < 500:
                raise ValueError(f'invalid EC sequence for head500: {entry}')
            fragment = sequence[:500]
            rows.append({'task': 'EC_level2', 'entry': entry, 'split': split_name, 'strategy': 'head500', 'window_length': 500, 'sequence_length': len(sequence), 'start_0based': 0, 'end_0based_exclusive': 500, 'start_1based': 1, 'end_1based_inclusive': 500, 'fragment_sha256': hashlib.sha256(fragment.encode()).hexdigest(), 'selection_uses_ground_truth': False, 'selection_rule': 'fixed_n_terminal_500'})
    frame = pd.DataFrame(rows)
    expected = sum((len(split[f'{name}_entries']) for name in ['train', 'val', 'test']))
    if len(frame) != expected or frame.entry.duplicated().any():
        raise RuntimeError('head500 manifest inventory mismatch')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + '.tmp.gz')
    frame.to_csv(tmp, index=False, compression='gzip')
    tmp.replace(OUT)
    print(f'[DONE] {OUT} rows={len(frame)}')
if __name__ == '__main__':
    main()
