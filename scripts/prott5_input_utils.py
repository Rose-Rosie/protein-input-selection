from project_paths import ROOT
import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

def load_prott5():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(model_name)
    if device.type == 'cpu':
        model.to(torch.float32)
    model = model.to(device)
    model = model.eval()
    return (tokenizer, model, device)

def clean_seq(seq: str) -> str:
    return re.sub('[UZOB]', 'X', str(seq).upper().strip())

def to_spaced(seq: str) -> str:
    return ' '.join(list(seq))

def embed_sequences_chunked(tokenizer, model, device, seqs: list[str], segment_batch_size: int) -> list[np.ndarray]:
    results = []
    for start in range(0, len(seqs), segment_batch_size):
        sub_seqs = [to_spaced(clean_seq(s)) for s in seqs[start:start + segment_batch_size]]
        ids = tokenizer(sub_seqs, return_tensors='pt', padding=True, add_special_tokens=True).to(device)
        with torch.no_grad():
            out = model(**ids).last_hidden_state
        for idx in range(len(sub_seqs)):
            mask = ids.attention_mask[idx].bool()
            valid = out[idx][mask][1:-1]
            results.append(valid.mean(dim=0).detach().cpu().numpy())
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
