# from tasks import feature_count
import pandas as pd
import tqdm
import os
import multiprocessing
import itertools
from sklearn import preprocessing
pre = os.getenv("HOME") + '/lnn-el/data/aida/template/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + dataset for dataset in datasets]

def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists)  
for pos, dataset in enumerate(datasets):
    df = pd.read_csv(dataset)
    feature_idx = df.columns.get_loc('Features')
    def process_prior(chunk):
        l, r = chunk
        dt = df.iloc[l: r].values.tolist()
        features = [eval(line[feature_idx]) for line in dt]
        priors =  [line[16] for line in features]
        normalized_priors = preprocessing.minmax_scale(priors, feature_range=(0.1, 1))#self.normalize(ref_scores)
        for i, _ in enumerate(features):
            features[i][7] = normalized_priors[i]
            dt[i][feature_idx] = str(features[i])
        # print(dt[0])
        return dt

    n, l, r = df.shape[0], 0, 0
    chunks = []
    pbar = tqdm.tqdm(total=len(set(df.QuestionMention)))
    # print(len(set(df.QuestionMention.values)))
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        chunks.append([l, r + 1])
        # lines = ''.join([f.readline() for i in range(r - l + 1)])
        # feature_count.apply_async([df.iloc[l: r + 1].to_json()])
        # feature_count(df.iloc[l: r + 1].to_json())
        l = r + 1
        r = l
        pbar.update(1)
    pbar.close()
    with multiprocessing.Pool(40) as pool:
        data = list(tqdm.tqdm(pool.map(process_prior, chunks), total = len(chunks), position=pos))
    data = list(flatten(data))
    # print(len(data[0]))
    new_df = pd.DataFrame(data, columns=['left','Mention_label','Features','Label','Mention','QuestionMention','db','blink','right', 'Question'])
    new_df.to_csv(dataset + '.2', index=False)