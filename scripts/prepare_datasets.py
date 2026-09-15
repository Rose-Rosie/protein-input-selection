"""Build three task datasets and accession-aligned random 80/10/10 splits."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from project_paths import ROOT,TASKS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--sequences',type=Path,required=True,help='Candidate CSV/TSV with Entry, Length, Sequence and task labels')
    p.add_argument('--labels',type=Path,help='Localization multi-hot CSV; EC/GO labels are read from --sequences')
    p.add_argument('--domains',type=Path,required=True,help='JSON: Entry -> one-based inclusive domain intervals')
    p.add_argument('--available-entries',type=Path,required=True,help='CSV with Entry, listing the frozen common representation inventory')
    a=p.parse_args()
    frame=pd.read_csv(a.sequences,sep='\t' if a.sequences.suffix=='.tsv' else ',')
    domains=json.loads(a.domains.read_text())
    allowed=set(pd.read_csv(a.available_entries)['Entry'].astype(str))
    frame=frame[frame.Length.between(1001,5000)&frame.Entry.isin(allowed)&frame.Entry.isin([k for k,v in domains.items() if v])].copy()
    if a.task=='Subcellular':
        if not a.labels:p.error('--labels is required for Subcellular')
        labels=pd.read_csv(a.labels);vocab=[c for c in labels if c!='Entry']
        frame=frame.merge(labels,on='Entry',validate='one_to_one')
    else:
        col='final_ec2' if a.task=='EC_level2' else 'final_labels'
        frame=frame.dropna(subset=[col])
        vocab=sorted({v for text in frame[col] for v in str(text).split(';') if v})
        for v in vocab:frame[v]=frame[col].map(lambda text:int(v in str(text).split(';')))
    frame=frame.sort_values('Entry').reset_index(drop=True)
    if frame.Entry.duplicated().any():raise ValueError('Accession identifiers must be unique within a task')
    frame['DomainCount']=frame.Entry.map(lambda e:len(domains[e]))
    for sub in ['data','labels','splits']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
    frame.to_csv(ROOT/'data'/f'{a.task}_master.csv',index=False)
    (ROOT/'data'/f'{a.task}_domains.json').write_text(json.dumps({e:domains[e] for e in frame.Entry}))
    frame[['Entry']+vocab].to_csv(ROOT/'labels'/f'{a.task}_labels_multihot.csv',index=False)
    pd.DataFrame({'Label':vocab}).to_csv(ROOT/'labels'/f'{a.task}_label_list.csv',index=False)
    tr,tmp=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42)
    va,te=train_test_split(tmp,test_size=.5,random_state=42)
    values={'label_cols':np.asarray(vocab),'random_state':np.asarray([42])}
    for split,idx in [('train',tr),('val',va),('test',te)]:
        values[f'{split}_entries']=frame.Entry.to_numpy()[idx]
        values[f'y_{split}']=frame[vocab].to_numpy(dtype=np.int32)[idx]
    np.savez_compressed(ROOT/'splits'/f'{a.task}_split_entries.npz',**values)
    frames=[pd.read_csv(ROOT/'data'/f'{t}_master.csv',usecols=['Entry','Length','Sequence']) for t in TASKS if (ROOT/'data'/f'{t}_master.csv').exists()]
    # Preserve the original source priority for accessions shared between tasks.
    priority=['Subcellular','EC_level2','GO_slim']
    frames=[pd.read_csv(ROOT/'data'/f'{t}_master.csv',usecols=['Entry','Length','Sequence']) for t in priority if (ROOT/'data'/f'{t}_master.csv').exists()]
    pd.concat(frames).drop_duplicates('Entry',keep='first').sort_values('Entry').to_csv(ROOT/'data/all_unique_full_length_entries.csv',index=False)
    print(a.task,len(tr),len(va),len(te),len(vocab))
if __name__=='__main__':main()
