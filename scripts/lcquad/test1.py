import pandas as pd
import Levenshtein
import numpy as np
import os
datasets = ['train', 'valid', 'test']
for name in datasets:
    # out = open(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv', 'w')
    # out.write('ID,Question,Classes,Entities\n')

    fname = f'../data/lcquad/blink_bert_box/full_{name}_gold.csv'
    df = pd.read_csv(fname)
    n, l, r = df.shape[0], 0, 0
    idx = 0
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        batch = df.iloc[l : r + 1]
        # lines = [f.readline() for _ in range(r - l + 1)]
        
        # try:
        gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
        if len(gold_pairs) != 1:
            print(len(gold_pairs), df.iloc[l]['QuestionMention'])
        # assert(0 == 1)
        idx+= 1
        l = r + 1
        r = l