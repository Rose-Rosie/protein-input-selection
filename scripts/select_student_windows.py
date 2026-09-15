"""Select windows from raw sequences using a trained student checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train_window_student import WindowStudentCNN, predict_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='CSV with Entry and Sequence')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = WindowStudentCNN(checkpoint['embedding_dim'], checkpoint['hidden_dim']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    window = int(checkpoint['window_length'])
    rows = []
    for row in pd.read_csv(args.input).itertuples(index=False):
        sequence = str(row.Sequence).upper().strip()
        if len(sequence) < window:
            raise ValueError(f'{row.Entry} has fewer than {window} residues')
        probability = predict_distribution(model, device, sequence, window)
        start = int(np.argmax(probability))
        rows.append({'Entry': row.Entry, 'start_0based': start,
                     'end_0based_exclusive': start + window,
                     'Sequence': sequence[start:start + window]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
