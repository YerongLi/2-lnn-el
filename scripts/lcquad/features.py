import pandas as pd
from tasks import modified_lev
datasets = ['train', 'valid', 'test']

for name in datasets:
    fname = f'../data/lcquad/blink_bert_box/{name}_gold.csv'
    df = pd.read_csv(fname)
    df.drop(['row_num','Unnamed: 0'], axis=1, errors='ignore', inplace=True)

    df.sort_values(by=['QuestionMention'])
    n, l, r = df.shape[0], 0, 0
    count = 0
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        append_feature.apply_async((df.iloc[l: r + 1].to_json(), name))
        l = r + 1
        r = l
