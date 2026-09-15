from project_paths import ROOT
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
import esm

def load_esm2(model_name: str):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Loading model: {model_name}')
    if model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f'fair-esm implementation only supports {DEFAULT_MODEL_NAME}, got {model_name}')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return (alphabet.get_batch_converter(), model, device)

def clean_seq(seq: str) -> str:
    return re.sub('[UZOB]', 'X', str(seq).upper().strip())

def embed_sequences_chunked(tokenizer, model, device, seqs: list[str], segment_batch_size: int) -> list[np.ndarray]:
    results = []
    for start in range(0, len(seqs), segment_batch_size):
        sub_seqs = [clean_seq(seq) for seq in seqs[start:start + segment_batch_size]]
        batch = [(str(idx), seq) for idx, seq in enumerate(sub_seqs)]
        _, _, tokens = tokenizer(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            outputs = model(tokens, repr_layers=[33], return_contacts=False)
            hidden = outputs['representations'][33]
        padding_idx = 1
        batch_lens = (tokens != padding_idx).sum(1)
        for idx in range(len(sub_seqs)):
            valid = hidden[idx, 1:batch_lens[idx] - 1]
            results.append(valid.mean(dim=0).detach().cpu().numpy().astype(np.float32))
    return results

def fixed_slice(seq: str, strategy: str) -> list[str]:
    seq = clean_seq(seq)
    length = len(seq)
    if length <= WINDOW_SIZE:
        return [seq]
    if strategy == 'head1000':
        return [seq[:WINDOW_SIZE]]
    if strategy == 'mid1000':
        start = max(0, (length - WINDOW_SIZE) // 2)
        return [seq[start:start + WINDOW_SIZE]]
    if strategy == 'tail1000':
        return [seq[-WINDOW_SIZE:]]
    if strategy == 'splice300_400_300':
        middle_start = max(0, (length - 400) // 2)
        return [seq[:300] + seq[middle_start:middle_start + 400] + seq[-300:]]
    if strategy == 'triple_mean':
        start = max(0, (length - WINDOW_SIZE) // 2)
        return [seq[:WINDOW_SIZE], seq[start:start + WINDOW_SIZE], seq[-WINDOW_SIZE:]]
    raise ValueError(f'Unsupported fixed strategy: {strategy}')

def sliding_windows(seq: str, strategy: str) -> list[str]:
    seq = clean_seq(seq)
    if strategy != 'no_overlap_window':
        raise ValueError(f'Unsupported sliding strategy: {strategy}')
    if len(seq) <= WINDOW_SIZE:
        return [seq]
    windows = []
    for start in range(0, len(seq), WINDOW_SIZE):
        window = seq[start:start + WINDOW_SIZE]
        if len(window) >= 100:
            windows.append(window)
    return windows or [seq[:WINDOW_SIZE]]

def select_domain_region(seq: str, domains: list[tuple[int, int]], strategy: str) -> Optional[str]:
    seq = clean_seq(seq)
    length = len(seq)
    if not domains:
        return None
    if length <= WINDOW_SIZE:
        return seq
    if strategy == 'domain_center_longest':
        longest = max(domains, key=lambda item: item[1] - item[0])
        center = (longest[0] + longest[1]) // 2
        start = max(0, center - WINDOW_SIZE // 2)
        end = min(length, start + WINDOW_SIZE)
        start = max(0, end - WINDOW_SIZE)
        return seq[start:end]
    if strategy == 'domain_max_cover':
        domain_centers = [(start + end) // 2 for start, end in domains]
        best_start, best_cover = (0, -1)
        for start in range(0, max(1, length - WINDOW_SIZE + 1), 200):
            end = start + WINDOW_SIZE
            cover = sum((start <= center <= end for center in domain_centers))
            if cover > best_cover:
                best_start, best_cover = (start, cover)
        return seq[best_start:best_start + WINDOW_SIZE]
    raise ValueError(f'Unsupported domain strategy: {strategy}')

WINDOW_SIZE = 1000
DEFAULT_MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
