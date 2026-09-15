"""Recalculate Macro-F1 from saved multi-label test predictions."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True,
                        help='NPZ with two-dimensional y_true and y_pred arrays')
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        truth, pred = data['y_true'], data['y_pred']
    if truth.ndim != 2 or truth.shape != pred.shape or truth.shape[1] == 0:
        raise ValueError('Expected equal two-dimensional multi-label arrays')
    if not np.isin(truth, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Targets and predictions must be binary')
    tp = ((truth == 1) & (pred == 1)).sum(axis=0)
    denominator = truth.sum(axis=0) + pred.sum(axis=0)
    scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    print(json.dumps({'proteins': len(truth), 'labels': truth.shape[1],
                      'macro_f1': float(scores.mean())}, indent=2))


if __name__ == '__main__':
    main()
