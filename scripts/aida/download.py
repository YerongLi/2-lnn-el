# from tasks import lookup, save_type
import pandas as pd
import argparse
from datetime import date, timedelta
import ast
from tqdm import tqdm
# from pymongo import MongoClient
from time import sleep
import requests
import traceback
import os

# sleep(0.05)

def lookup(query):
	link = f'https://lookup.dbpedia.org/api/search/KeywordSearch?QueryString=%22{query}%22&MaxHits=800'
	filename = f'./candidates/{query}.xml'
	try:
		# if not os.path.isfile(filename):
		if True:
			r = requests.get(link)
			open(filename, 'wb').write(r.content)
	except Exception as e:
		traceback.print_exc()

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
		datasets = ['aida_train.txt_groundtruth.csv', 'temp_aida_dev_groundtruth.csv', 'temp_aida_test_groundtruth.csv']
		for name in datasets:
			csvfile = pd.read_csv(name)
			s = set(pair for pair in csvfile['Mention'])
			for ipt in tqdm(s):
				# lookup.apply_async([ipt])
				filename = f'./candidates/{ipt}.xml'
				# if not os.path.isfile(filename):
				if True:
					lookup(ipt)
				# sleep(0.05)

save_xml(saveType=False, saveLookup=True)