"""Encode fixed-position and domain-centred inputs using the manuscript pipelines."""
import argparse
import json
import numpy as np
import pandas as pd
from project_paths import ROOT,TASKS
STRATEGIES=['head1000','mid1000','tail1000','splice300_400_300','domain_center_longest','domain_max_cover']

def select_segments(sequence,domains,strategy,model,task,utils):
    if not strategy.startswith('domain_'):return utils.fixed_slice(sequence,strategy)
    if model=='prott5' and task=='Subcellular' and strategy=='domain_center_longest':
        # Supplementary Methods S2: this source pipeline did not back-shift at the C terminus.
        longest=max(domains,key=lambda x:x[1]-x[0]);center=(longest[0]+longest[1])//2
        start=max(0,center-500)
        return [utils.clean_seq(sequence)[start:min(len(sequence),start+1000)]]
    return [utils.select_domain_region(sequence,domains,strategy=strategy)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['esm2','prott5'],required=True)
    p.add_argument('--tasks',nargs='+',choices=TASKS,default=TASKS)
    p.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=STRATEGIES)
    p.add_argument('--batch-size',type=int,default=1)
    args=p.parse_args()
    if args.model=='esm2':
        import esm2_input_utils as utils
        tokenizer,model,device=utils.load_esm2(utils.DEFAULT_MODEL_NAME)
    else:
        import prott5_input_utils as utils
        tokenizer,model,device=utils.load_prott5()
    for task in args.tasks:
        frame=pd.read_csv(ROOT/'data'/f'{task}_master.csv')
        domains=json.loads((ROOT/'data'/f'{task}_domains.json').read_text())
        out=ROOT/'embeddings_1000aa'/args.model/task;out.mkdir(parents=True,exist_ok=True)
        for strategy in args.strategies:
            values={}
            for row in frame.itertuples(index=False):
                segments=select_segments(row.Sequence,domains[row.Entry],strategy,args.model,task,utils)
                vectors=utils.embed_sequences_chunked(tokenizer,model,device,segments,args.batch_size)
                values[row.Entry]=np.mean(vectors,axis=0)
            np.savez(out/f'{strategy}.npz',**values)
if __name__=='__main__':main()
