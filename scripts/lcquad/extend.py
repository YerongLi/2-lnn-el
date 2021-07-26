import pandas as pd
import tqdm
datasets = ['train', 'valid', 'test']
target_length = 20
def ext(features):
    features = eval(features)
    features.extend((target_length - len(features)) *[0.0])
    return str(features)
for name in datasets:
    df = pd.read_csv(f'../data/lcquad/blink_bert_box/full_{name}_gold.csv')
    df['Features'] = df['Features'].apply(ext)
    # for i in tqdm.tqdm(range(df.shape[0])):
    #     features = eval(df.iloc[i]['Features'])
    #     l = len(features)
    #     features.extend([0.0]* (target_length - l))

        # df.iloc[i, df.columns.get_loc('Features')] = str(features)
    df.to_csv(f'../data/lcquad/blink_bert_box/extend_{name}_gold.csv')