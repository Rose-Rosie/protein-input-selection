"""Merge a UniProt-to-UniRef50 mapping and retain the first entry per mapped cluster."""
import argparse
from pathlib import Path
import pandas as pd
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--mapping',type=Path,required=True,help='Two-column UniProt ID mapping TSV: From, UniRef50')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    raw=pd.read_csv(a.input,sep='\t')
    mapping=pd.read_csv(a.mapping,sep='\t')
    if len(mapping.columns)!=2:raise ValueError('Expected exactly two mapping columns')
    mapping.columns=['Entry','UniRef50_ID']
    merged=raw.merge(mapping,on='Entry',how='left')
    mapped=merged[merged.UniRef50_ID.notna()].drop_duplicates('UniRef50_ID',keep='first')
    unmapped=merged[merged.UniRef50_ID.isna()]
    result=pd.concat([mapped,unmapped],ignore_index=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,sep='\t',index=False)
    print(f'Input {len(raw)}; retained {len(result)}; unmapped {len(unmapped)}')
if __name__=='__main__':main()
