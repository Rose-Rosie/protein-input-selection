"""Intersect accession keys from the representation files used to define an analysis set."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
def main():
    p=argparse.ArgumentParser();p.add_argument('--embeddings',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    shared=None
    for path in a.embeddings:
        with np.load(path,allow_pickle=False) as z:keys={k for k in z.files if not k.startswith('__')}
        shared=keys if shared is None else shared&keys
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'Entry':sorted(shared)}).to_csv(a.output,index=False)
if __name__=='__main__':main()
