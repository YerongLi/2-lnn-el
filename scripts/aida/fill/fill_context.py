import pymongo
import pandas as pd
import tqdm

client = pymongo.MongoClient(host='localhost', port=27017)
db = client.dbpedia

features = db['features']
pre = '../'
datasets = ['aida_train.txt_data.csv', 'temp_aida_dev_data.csv', 'temp_aida_test_data.csv']
datasets = [pre + dataset for dataset in datasets]

json = list(features.find({'t_' : 'c1'}))
djson = dict()
for item in json:
	if item['Q'] not in djson:
		djson[item['Q']] = dict()
	if item['men'] not in djson[item['Q']]:
		djson[item['Q']][item['men']] = dict()
	# if item['cand'] not in djson[item['Q']][item['men']]:
	djson[item['Q']][item['men']][item['cand']] = item['s_']

for dataset in datasets[:1]:
	df = pd.read_csv(dataset)
	feature_idx = df.columns.get_loc('Features')
	n = df.shape[0]
	for i in tqdm.tqdm(range(n)):
		try:
			mention, candidate = df.iloc[i].Mention_label.split(';')
			question = df.iloc[i].Question
		except:
			with open('error.txt', 'a') as f:
				f.write(df.iloc[i].Mention_label + '\n')
				print(df.iloc[i].Mention_label.split(';'))
			continue
		try:
			res = djson[question][mention][candidate]
		except:
			with open('error.txt', 'a') as f:
				f.write(question + '\n')
				f.write(mention + '\n')
				f.write(candidate + '\n')
				print(question, mention, candidate)
			continue	
		feature_v = eval(df.iloc[i].Features)
		feature_v[8] = res
		df.iloc[i, feature_idx] = str(feature_v)
	print(df.Features.values)
	# df.to_csv(dataset, index=False)


