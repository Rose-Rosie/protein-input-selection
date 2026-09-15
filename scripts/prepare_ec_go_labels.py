import pandas as pd
import re
from goatools.obo_parser import GODag
from collections import Counter
import os
import argparse
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--go-obo', required=True)
p.add_argument('--slim-obo', required=True)
p.add_argument('--outdir', required=True)
args = p.parse_args()
Path(args.outdir).mkdir(parents=True, exist_ok=True)
INPUT_FILE = args.input
GO_OBO = args.go_obo
SLIM_OBO = args.slim_obo
OUTPUT_FILE = str(Path(args.outdir) / 'final_slim_dataset.csv')
TOP_N_TO_KEEP = 30
if not all((os.path.exists(f) for f in [INPUT_FILE, GO_OBO, SLIM_OBO])):
    print('❌ 请确保输入文件、go-basic.obo 和 goslim_generic.obo 存在')
    exit()
print('加载 GO DAG...')
go_dag = GODag(GO_OBO)
print('加载 GO Slim DAG...')
slim_dag = GODag(SLIM_OBO)
SLIM_SET = set(slim_dag.keys())
print('Slim GO数量:', len(SLIM_SET))
overlap = set(go_dag.keys()) & SLIM_SET
print('GO与Slim交集:', len(overlap))

def extract_go_terms(go_str):
    if pd.isna(go_str):
        return []
    return re.findall('GO:\\d+', str(go_str))

def map_to_slim(go_str):
    gos = extract_go_terms(go_str)
    mapped = set()
    for go_id in gos:
        if go_id not in go_dag:
            continue
        ancestors = go_dag[go_id].get_all_parents()
        ancestors.add(go_id)
        mapped.update(ancestors & SLIM_SET)
    return mapped
print('读取数据...')
df = pd.read_csv(INPUT_FILE, sep='\t')
go_col = 'Gene Ontology IDs' if 'Gene Ontology IDs' in df.columns else 'go_id'
print('GO列:', go_col)
print('执行GO → Slim映射...')
df['mapped_slims'] = df[go_col].apply(map_to_slim)
total = len(df)
non_empty = sum((len(x) > 0 for x in df['mapped_slims']))
print(f'非空mapping: {non_empty}/{total} ({non_empty / total:.2%})')
root_nodes = {'GO:0008150', 'GO:0003674', 'GO:0005575'}
all_labels = [label for sublist in df['mapped_slims'] for label in sublist if label not in root_nodes]
label_counts = Counter(all_labels)
count_df = pd.DataFrame(label_counts.most_common(), columns=['GO_ID', 'Count'])
count_df['Label_Name'] = count_df['GO_ID'].apply(lambda x: go_dag[x].name if x in go_dag else 'Unknown')
count_df['Cumulative_Percentage'] = count_df['Count'].cumsum() / count_df['Count'].sum() * 100
print('\nTop 10 labels:')
print(count_df.head(10))
threshold = 80
selected = count_df[count_df['Cumulative_Percentage'] <= threshold]
target_labels = set(selected['GO_ID'])
len(target_labels)
df['label_count'] = df['mapped_slims'].apply(len)
print('\nLabel density:')
print(df['label_count'].describe())
df.head()
print('\n构建最终数据集...')

def filter_labels(slim_set):
    valid = slim_set & target_labels
    return ';'.join(valid) if valid else None
df['final_labels'] = df['mapped_slims'].apply(filter_labels)
df_final = df.dropna(subset=['final_labels']).copy()
output_cols = ['Entry', 'Sequence', 'Length', 'final_labels']
output_cols = [c for c in output_cols if c in df_final.columns]
df_final.head()
df_final.isna().sum()
df_final.shape
output_cols
df_final[output_cols].to_csv(OUTPUT_FILE, index=False)
print(f'\n✅ 完成！')
print(f'最终样本数: {len(df_final)}')
print(f'保存路径: {OUTPUT_FILE}')
ec_col = 'EC number'
ec_col = 'EC number' if 'EC number' in df_final.columns else [c for c in df_final.columns if 'ec' in c.lower()][0]
print('EC column:', ec_col)
import re
import pandas as pd

def extract_ec_level2(ec_str):
    if pd.isna(ec_str):
        return []
    ecs = re.findall('\\d+\\.\\d+\\.\\d+\\.\\d+|\\d+\\.\\d+\\.\\d+\\.-|\\d+\\.\\d+\\.-\\.-|\\d+\\.-\\.-\\.-', str(ec_str))
    level2 = []
    for ec in ecs:
        parts = ec.split('.')
        if len(parts) >= 2:
            level2.append(f'{parts[0]}.{parts[1]}')
    return list(set(level2))
df_final['ec_level2'] = df_final[ec_col].apply(extract_ec_level2)
from collections import Counter
all_ec2 = [ec for sub in df_final['ec_level2'] for ec in sub]
ec2_counts = Counter(all_ec2)
ec2_df = pd.DataFrame(ec2_counts.most_common(), columns=['EC2', 'Count'])
ec2_df.head(46)
len(ec2_df)
TOP_N = 10
target_ec2 = set(ec2_df.head(TOP_N)['EC2'])

def filter_ec2(ec_list):
    valid = set(ec_list) & target_ec2
    return ';'.join(valid) if valid else None
df_final['final_ec2'] = df_final['ec_level2'].apply(filter_ec2)
df_ec2_final = df_final.dropna(subset=['final_ec2']).copy()
output_cols = ['Entry', 'Sequence', 'Length', 'final_ec2']
df_ec2_final[output_cols].to_csv(Path(args.outdir) / 'ec_level2_dataset.csv', index=False)
df_ec2_final.shape
set(ec2_df.head(TOP_N)['EC2'])
df_ec2_final.head()
