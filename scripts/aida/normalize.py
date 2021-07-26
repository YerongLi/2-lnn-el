import os
import multiprocessing
import pandas as pd
import tqdm
pre = os.getenv('HOME') + '/2-lnn-el/data/aida/blink_bert_box/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [str(i) + ';' + pre + d for i, d in enumerate(datasets)]
def process(doc):
	i, doc = doc.split(';')
	df = pd.read_csv(doc)
	for _ in tqdm.tqdm(range(df.shape[0]), position=i):
		pass
with multiprocessing.Pool(3) as p:
	p.map(process, datasets)