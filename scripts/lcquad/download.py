from tasks import lookup, save_type
import pandas as pd
import argparse
from datetime import date, timedelta
import ast
from tqdm import tqdm
from pymongo import MongoClient
from time import sleep
sleep(0.05)



def save_xml(saveType = True, saveLookup = False):
	"""
	获得A股所有股票日线行情数据

	Args:
		start_date (str): 起始日
		end_date (str): 结束日
	"""
	if saveType:
		# benchmark = pd.read_csv(f'../data/lcquad/full_lcquad_gt_5000.csv')
		benchmark = pd.read_csv(f'../data/lcquad/full_lcquad_gt_5000.csv')

		benchmark['Entities'] = benchmark['Entities'].astype(object)

		for _ , gold_row in tqdm(benchmark.iterrows()):
			gold_entities = ast.literal_eval(gold_row['Entities'])
			for entity in gold_entities:
				save_type.apply_async([entity.replace('\t', "'")])

	if saveLookup:
		datasets = ['train', 'valid', 'test']
		for name in datasets:
			# csvfile = pd.read_csv(f'../data/lcquad/blink_bert_box/{name}_gold.csv')
			# s = set(str(pair).split(';')[0] for pair in csvfile['Mention_label'])

			for ipt in tqdm(s):
				# lookup.apply_async([ipt])
				lookup(ipt)
				sleep(0.05)
			break

save_xml(saveType=True, saveLookup=False)