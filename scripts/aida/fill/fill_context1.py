import multiprocessing
from operator import mul
import pymongo
import pandas as pd
import tqdm
import os
pbar = tqdm.tqdm()
client = pymongo.MongoClient(host='localhost', port=27017)
db = client.dbpedia

logname = os.path.basename(__file__) + '.log'

if os.path.exists(logname):
  os.remove(logname)

features = db['features']
pre = os.getenv("HOME") + '/lnn-el/data/aida/template/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + dataset for dataset in datasets]

json_list = list(features.find({'t_' : 'c1'}))
djson = dict()
for item in json_list:
	if item['d'] not in djson:
		djson[item['d']] = dict()
	if item['men'] not in djson[item['d']]:
		djson[item['d']][item['men']] = dict()
	if item['cand'] not in djson[item['d']][item['men']]:
		djson[item['d']][item['men']][item['cand']] = item['s_']
del json_list


for pos, dataset in enumerate(datasets):
	df = pd.read_csv(dataset)
	feature_idx = df.columns.get_loc('Features')
	n = df.shape[0]
	def process(i):
		try:
			data = df.iloc[i].values
			mention, candidate = df.iloc[i].Mention_label.split('===')
			doc = str(df.iloc[i].Question)
			if doc in djson and mention in djson[doc] and candidate in djson[doc][mention]:
				res = djson[doc][mention][candidate]
		except:
			import traceback
			traceback.print_exc()
			with open(logname, 'a') as f:
				f.write(df.iloc[i].Mention_label + '\n')
				print(df.iloc[i].Mention_label.split('==='))
			return data

		feature_v = eval(df.iloc[i].Features)
		feature_v[8] = res
		feature_v[17] = res
		data[feature_idx] = str(feature_v)
		pbar.update(1)
		return data


	with multiprocessing.Pool(40) as pool:
		data = pool.map(process, range(n))
	# print(data)
	new_df = pd.DataFrame(data, columns=['left','Mention_label','Features','Label','Mention','QuestionMention','db','blink','right', 'Question'])

	new_df.to_csv(dataset + '.2', index=False)
