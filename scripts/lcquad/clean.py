import pandas as pd
import Levenshtein
import numpy as np
import os
datasets = ['train', 'valid', 'test']
# datasets = ['valid', 'test']
# datasets = ['test']
for name in datasets:
    fname = f'../data/lcquad/blink_bert_box/{name}_gold.csv'
    df = pd.read_csv(fname)
    df.sort_values(by=['QuestionMention', 'Label'])
    df.to_csv(fname + '.1', index = False)

for name in datasets:
    # out = open(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv', 'w')
    # out.write('ID,Question,Classes,Entities\n')
    out = open(f'../data/lcquad/blink_bert_box/full_{name}_gold.csv', 'w')

    fname = f'../data/lcquad/blink_bert_box/{name}_gold.csv' + '.1'
    f = open(fname, "r")
    df = pd.read_csv(fname)
    out.write(f.readline())
    n, l, r = df.shape[0], 0, 0
    idx = 0
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        batch = df.iloc[l : r + 1]
        lines = [f.readline() for _ in range(r - l + 1)]
        
        try:
            gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
            mention = gold_pairs[0].split(';')[0]
            assert(len(lines) == batch.shape[0])
            scores = np.array([Levenshtein.ratio(mention, pairs.split(';')[1]) for pairs in gold_pairs])
            j = np.argmax(scores)
            out.write(''.join(lines[: -len(gold_pairs)]))
            x = lines[-len(gold_pairs) + j]
            if df.iloc[l]['QuestionMention'] == 'Which spouse of Ptolemy XIV had a mother named Ptolemy XII auletes ?--Ptolemy XIV':
                print(len(lines))
                # print(lines[0])
                print(batch)
                print(scores)
                print(j)
                print(x)
            out.write(x)
        except:
            import traceback
            print(df.iloc[l].QuestionMention)
            # print(gold_pairs)
            # traceback.print_exc()

        idx+= 1
        l = r + 1
        r = l
    os.remove(fname)
    print(f'{name} Finished.')