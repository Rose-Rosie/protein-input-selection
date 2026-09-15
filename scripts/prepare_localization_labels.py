"""Parse localization names and retain labels present in more than 100 candidate entries."""
import argparse,re
from collections import Counter
from pathlib import Path
import pandas as pd
def data_process(note: str) -> set:
    """
    提取 UniProt 亚细胞定位文本中每个子句的第一个单词并去重。

    参数
    ----
    note : str
        原始文本，例如 SUBCELLULAR LOCATION: ... Note=...

    返回
    ----
    set
        去重后的第一个单词集合。
    """
    text = note.split('Note=')[0]
    text = re.sub('\\s*SUBCELLULAR LOCATION:\\s*', '', text, flags=re.I)
    text = re.sub('\\{[^}]*\\}', '', text)
    text = re.sub('\\([^)]*\\)', '', text)
    text = re.sub('\\[[^\\]]*\\]:\\s*', '', text)
    text = re.sub('\\[[^\\]]*\\]', '', text)
    text = re.sub('\\r\\n+', ' ', text)
    text = re.sub('\\s+', ' ', text)
    text = [s.strip() for s in re.split('\\.\\s*', text) if s.strip()]
    text = {re.split('[;,]', s, maxsplit=1)[0].strip() for s in text}
    text = {i for i in text if i}
    return text

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    frame=pd.read_csv(a.input,sep='\t' if a.input.suffix=='.tsv' else ',')
    col='Subcellular location [CC]'
    frame=frame[(frame.Length>1000)&frame[col].notna()].copy()
    frame=frame[~frame[col].str.contains('isoform',case=False)].copy()
    parsed=frame[col].map(data_process)
    counts=Counter(label for labels in parsed for label in labels)
    vocabulary=sorted(label for label,count in counts.items() if count>100)
    result=pd.DataFrame({'Entry':frame.Entry})
    for label in vocabulary:result[label]=parsed.map(lambda labels:int(label in labels))
    result=result[result[vocabulary].sum(axis=1)>0]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print(f'{len(result)} entries, {len(vocabulary)} labels')
if __name__=='__main__':main()
