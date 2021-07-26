import pandas as pd
category = 'Scientist'
datasets = ['valid', 'test']
out = open(f'../data/lcquad/{category}_lcquad_gt_5000.csv', 'w')
out.write('ID,Question,Classes,Entities\n')

for name in datasets:
    # out = open(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv', 'w')
    # out.write('ID,Question,Classes,Entities\n')

    df = pd.read_csv(f'../data/lcquad/blink_bert_box/{category}_{name}.csv')
    n, l, r = df.shape[0], 0, 0
    idx = 0
    while r < n:
        while r < n - 1 and df.iloc[l]['Question'] == df.iloc[r + 1]['Question']:
            r+= 1
        batch = df.iloc[l : r + 1]
        gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
        gold_entities = set([pair.split(';')[-1].replace(' ', '_') for pair in gold_pairs if not "'" in pair])
        line = f'{idx},\"{df.iloc[l].Question}\",[],\"{str(gold_entities)}\"\n'
        out.write(line)
        idx+= 1
        l = r + 1
        r = l
    # out.close()
    # _ = pd.read_csv(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
    # print(f'[Tested] ../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
out.close()
_ = pd.read_csv(f'../data/lcquad/{category}_lcquad_gt_5000.csv')
print(f'[Tested] ../data/lcquad/{category}_lcquad_gt_5000.csv')
