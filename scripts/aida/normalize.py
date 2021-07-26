import os
import multiprocessing
import pandas as pd
pre = os.getenv('HOME') + '/2-lnn-el/data/blink_bert_box/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + d for d in datasets]
def process(doc):
	pd.read_csv(doc)
	pass
with multiprocessing.Pool(3) as p:
	p.map(process, datasets)