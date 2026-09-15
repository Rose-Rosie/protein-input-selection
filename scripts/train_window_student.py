"""Distil cross-fitted teacher window distributions into a lightweight CNN."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from project_paths import ROOT
TASKS = ['Subcellular', 'EC_level2', 'GO_slim']
RANDOM_SEEDS = [101, 102, 103, 104, 105]
AA = 'ACDEFGHIKLMNPQRSTVWYX'
TOKEN = {aa: idx + 1 for idx, aa in enumerate(AA)}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--window-length', type=int, default=500)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--emd-weight', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--teacher-root', type=Path, default=ROOT / 'analysis_outputs' / 'window_distillation' / 'teacher')
    parser.add_argument('--outdir', type=Path, default=ROOT / 'analysis_outputs' / 'window_distillation' / 'student')
    return parser.parse_args()

class SequenceWindowDataset(Dataset):

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]

def collate(records: list[dict]) -> dict:
    max_length = max((len(row['tokens']) for row in records))
    tokens = torch.zeros((len(records), max_length), dtype=torch.long)
    lengths = []
    for idx, row in enumerate(records):
        length = len(row['tokens'])
        tokens[idx, :length] = torch.from_numpy(row['tokens'])
        lengths.append(length)
    return {'tokens': tokens, 'lengths': lengths, 'targets': [torch.from_numpy(row['target']) for row in records], 'entries': [row['entry'] for row in records]}

class WindowStudentCNN(nn.Module):
    """Small sequence-only CNN that emits one relevance logit per residue."""

    def __init__(self, embedding_dim: int=32, hidden_dim: int=64):
        super().__init__()
        self.embedding = nn.Embedding(len(TOKEN) + 1, embedding_dim, padding_idx=0)
        self.network = nn.Sequential(nn.Conv1d(embedding_dim, hidden_dim, kernel_size=9, padding=4), nn.GELU(), nn.Conv1d(hidden_dim, hidden_dim, kernel_size=33, padding=16, groups=hidden_dim), nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1), nn.GELU(), nn.Conv1d(hidden_dim, 1, kernel_size=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(tokens).transpose(1, 2)).squeeze(1)

def encode_sequence(sequence: str) -> np.ndarray:
    cleaned = str(sequence).upper().strip().replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    return np.asarray([TOKEN.get(aa, TOKEN['X']) for aa in cleaned], dtype=np.int64)

def load_teacher_records(task: str, root: Path, sequences: dict[str, str]) -> tuple[list[dict], pd.DataFrame]:
    manifest = pd.read_csv(root / f'{task}__teacher_manifest.csv.gz', dtype={'entry': str})
    with np.load(root / f'{task}__teacher_distribution.npz') as data:
        entries = data['entries'].astype(str)
        offsets = data['offsets'].astype(int)
        probability = data['probability'].astype(np.float32)
    if not np.array_equal(entries, manifest['entry'].to_numpy(str)):
        raise RuntimeError(f'teacher manifest/distribution order mismatch for {task}')
    records = []
    for idx, row in enumerate(manifest.itertuples(index=False)):
        target = probability[offsets[idx]:offsets[idx + 1]].copy()
        records.append({'entry': str(row.entry), 'split': str(row.split), 'tokens': encode_sequence(sequences[str(row.entry)]), 'target': target})
    return (records, manifest)

def distribution_loss(residue_logits: torch.Tensor, lengths: list[int], targets: list[torch.Tensor], window_length: int, emd_weight: float) -> tuple[torch.Tensor, float, float]:
    losses = []
    cross_entropy_values, emd_values = ([], [])
    for idx, length in enumerate(lengths):
        window_logits = F.avg_pool1d(residue_logits[idx, :length][None, None, :], window_length, stride=1).flatten()
        target = targets[idx].to(window_logits.device)
        if len(window_logits) != len(target):
            raise RuntimeError('student/teacher candidate-window mismatch')
        log_probability = F.log_softmax(window_logits, dim=0)
        probability = log_probability.exp()
        cross_entropy = -(target * log_probability).sum()
        emd = torch.mean(torch.abs(torch.cumsum(probability, 0) - torch.cumsum(target, 0)))
        losses.append(cross_entropy + emd_weight * emd)
        cross_entropy_values.append(float(cross_entropy.detach()))
        emd_values.append(float(emd.detach()))
    return (torch.stack(losses).mean(), float(np.mean(cross_entropy_values)), float(np.mean(emd_values)))

def run_epoch(model, loader, device, optimizer, args) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, total_ce, total_emd, batches = (0.0, 0.0, 0.0, 0)
    for batch in loader:
        tokens = batch['tokens'].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            scores = model(tokens)
            loss, ce, emd = distribution_loss(scores, batch['lengths'], batch['targets'], args.window_length, args.emd_weight)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.detach())
        total_ce += ce
        total_emd += emd
        batches += 1
    return {'loss': total_loss / batches, 'cross_entropy': total_ce / batches, 'emd': total_emd / batches}

def predict_distribution(model, device, sequence: str, window_length: int) -> np.ndarray:
    tokens = torch.from_numpy(encode_sequence(sequence))[None, :].to(device)
    with torch.inference_mode():
        residue_logits = model(tokens)[0]
        logits = F.avg_pool1d(residue_logits[None, None, :], window_length, stride=1).flatten()
        return torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)

def fragment_sha(sequence: str, start: int, length: int) -> str:
    return hashlib.sha256(sequence[start:start + length].encode('ascii')).hexdigest()

def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    if args.require_gpu and (not torch.cuda.is_available()):
        raise RuntimeError('GPU is required but CUDA is unavailable')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.outdir.mkdir(parents=True, exist_ok=True)
    sequence_frame = pd.read_csv(ROOT / 'data' / 'all_unique_full_length_entries.csv', usecols=['Entry', 'Sequence'], dtype={'Entry': str, 'Sequence': str})
    sequences = dict(zip(sequence_frame.Entry, sequence_frame.Sequence))
    coordinate_rows: list[dict] = []
    fidelity_rows: list[dict] = []
    history_rows: list[dict] = []
    for task in args.tasks:
        records, teacher_manifest = load_teacher_records(task, args.teacher_root, sequences)
        train_records = [row for row in records if row['split'] == 'train']
        val_records = [row for row in records if row['split'] == 'val']
        train_loader = DataLoader(SequenceWindowDataset(train_records), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        val_loader = DataLoader(SequenceWindowDataset(val_records), batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
        model = WindowStudentCNN(args.embedding_dim, args.hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
        best_loss, best_state, stale = (float('inf'), None, 0)
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer, args)
            val_metrics = run_epoch(model, val_loader, device, None, args)
            history_rows.append({'task': task, 'epoch': epoch, **{f'train_{k}': v for k, v in train_metrics.items()}, **{f'val_{k}': v for k, v in val_metrics.items()}})
            print(f"[STUDENT] task={task} epoch={epoch} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f}", flush=True)
            if val_metrics['loss'] < best_loss - 1e-06:
                best_loss = val_metrics['loss']
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
        if best_state is None:
            raise RuntimeError(f'student failed to produce checkpoint for {task}')
        model.load_state_dict(best_state)
        checkpoint = {'state_dict': best_state, 'task': task, 'window_length': args.window_length, 'embedding_dim': args.embedding_dim, 'hidden_dim': args.hidden_dim, 'vocabulary': TOKEN, 'best_validation_loss': best_loss}
        torch.save(checkpoint, args.outdir / f'{task}__student.pt')
        teacher_by_entry = teacher_manifest.set_index('entry')
        record_by_entry = {row['entry']: row for row in records}
        for index, entry in enumerate(teacher_manifest['entry'].astype(str), start=1):
            meta = teacher_by_entry.loc[entry]
            sequence = str(sequences[entry]).upper().strip()
            selection_started = time.perf_counter()
            student_probability = predict_distribution(model, device, sequence, args.window_length)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            selection_seconds = time.perf_counter() - selection_started
            teacher_probability = record_by_entry[entry]['target']
            student_start = int(np.argmax(student_probability))
            teacher_start = int(meta['teacher_start_0based'])
            overlap = max(0, args.window_length - abs(student_start - teacher_start))
            fidelity_rows.append({'task': task, 'entry': entry, 'split': str(meta['split']), 'student_teacher_js': float(0.5 * (np.sum(student_probability * np.log(np.maximum(student_probability, 1e-30) / np.maximum(0.5 * (student_probability + teacher_probability), 1e-30))) + np.sum(teacher_probability * np.log(np.maximum(teacher_probability, 1e-30) / np.maximum(0.5 * (student_probability + teacher_probability), 1e-30))))), 'start_absolute_error': abs(student_start - teacher_start), 'window_overlap_fraction': overlap / args.window_length, 'teacher_probability_at_student_start': float(teacher_probability[student_start]), 'student_selection_seconds': selection_seconds})
            starts = {'student_window': student_start, 'teacher_window': teacher_start}
            for seed in RANDOM_SEEDS:
                starts[f'random_seed{seed}'] = int(meta[f'random_seed{seed}_start_0based'])
            for strategy, start in starts.items():
                coordinate_rows.append({'task': task, 'entry': entry, 'split': str(meta['split']), 'strategy': strategy, 'sequence_length': len(sequence), 'window_length': args.window_length, 'start_0based': start, 'end_0based_exclusive': start + args.window_length, 'fragment_sha256': fragment_sha(sequence, start, args.window_length), 'selection_uses_entry_label': False, 'selection_uses_test_labels': False, 'student_input': 'raw_amino_acid_sequence_only' if strategy == 'student_window' else 'not_applicable'})
            if index % 250 == 0 or index == len(teacher_manifest):
                print(f'[SELECT] task={task} {index}/{len(teacher_manifest)}', flush=True)
    coordinates = pd.DataFrame(coordinate_rows).sort_values(['task', 'strategy', 'split', 'entry'], ignore_index=True)
    coordinates.to_csv(args.outdir / 'selected_window_manifest.csv.gz', index=False)
    fidelity = pd.DataFrame(fidelity_rows)
    fidelity.to_csv(args.outdir / 'student_teacher_fidelity_per_protein.csv.gz', index=False)
    fidelity.groupby(['task', 'split'], sort=True).agg(proteins=('entry', 'size'), mean_js=('student_teacher_js', 'mean'), mean_start_absolute_error=('start_absolute_error', 'mean'), mean_window_overlap_fraction=('window_overlap_fraction', 'mean')).reset_index().to_csv(args.outdir / 'student_teacher_fidelity_summary.csv', index=False)
    pd.DataFrame(history_rows).to_csv(args.outdir / 'student_training_history.csv', index=False)
    parameter_counts = {}
    for task in args.tasks:
        checkpoint = torch.load(args.outdir / f'{task}__student.pt', map_location='cpu')
        parameter_counts[task] = int(sum((value.numel() for value in checkpoint['state_dict'].values())))
    audit = {'tasks': args.tasks, 'window_length': args.window_length, 'student_input': 'raw_amino_acid_sequence_only', 'teacher_targets': 'cross_fitted_continuous_window_distributions', 'test_labels_used': False, 'strategies': ['student_window', 'teacher_window', *[f'random_seed{s}' for s in RANDOM_SEEDS]], 'student_parameters': parameter_counts, 'coordinate_rows': len(coordinates)}
    (args.outdir / 'student_audit.json').write_text(json.dumps(audit, indent=2))
    print(f'[DONE] student distillation coordinates={len(coordinates)}', flush=True)
if __name__ == '__main__':
    main()
