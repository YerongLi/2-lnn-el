import spacy
import py_stringmatching as sm
import pandas as pd
import pymongo
import multiprocessing
import tqdm
import sys
import os
import concurrent.futures

logname = os.path.basename(__file__) + '.log'

if os.path.exists(logname):
  os.remove(logname)

manager = multiprocessing.Manager()
# # 建立mongodb连接
client = pymongo.MongoClient(host='localhost', port=27017)

# # 连接stock数据库，注意只有往数据库中插入了数据，数据库才会自动创建
db = client.dbpedia

# # 创建一个daily集合，类似于MySQL中“表”的概念
# group = db['group']

features = db['features']
feature_list = list(features.find({'t_': 'sim'}))
feature_dict = manager.list()
spacy_nlp = spacy.load("en_core_web_lg")
class Features:
	jw = sm.similarity_measure.jaro_winkler.JaroWinkler()
	lev = sm.similarity_measure.levenshtein.Levenshtein()
	smith_waterman = sm.similarity_measure.smith_waterman.SmithWaterman()
	partial_ratio = sm.similarity_measure.partial_ratio.PartialRatio()
	jac = sm.similarity_measure.jaccard.Jaccard()
print('Caching the features database')
def cacheEntry(entry):
	global known_feature_dict
	known_feature_dict.add((entry['men'], entry['cand']))
# feature_list = feature_list

# with multiprocessing.Pool(10) as pool:
# 	[ _ for _ in tqdm.tqdm(pool.imap_unordered(cacheEntry, feature_list), total=len(feature_list))]
known_feature_dict = set()
for entry in  tqdm.tqdm(feature_list):
	known_feature_dict.add((entry['men'], entry['cand']))
# known_feature_dict = known_feature_dict)

del feature_list
# sys.exit()
def save():
	print('Saving the dicitonary to database')
	features.insert_many(list(feature_dict))
	print(f'Saved {len(feature_dict)} Records.')
def stringsim(mentionCand):
	global feature_dict
	try:
		mention, candidate = mentionCand[0], mentionCand[1]
		if (mention, candidate) not in known_feature_dict:
			# print(mention, candidate)
			scores = dict()
			scores['jw']= Features.jw.get_sim_score(mention,candidate)
			scores['jac'] = Features.jac.get_sim_score(mention.split(' '),candidate.split(' '))
			scores['lev'] = Features.lev.get_sim_score(mention,candidate)
			scores['spacy'] = spacy_nlp(mention).similarity(spacy_nlp(candidate))
			feature_dict.append({'cand' :candidate, 'men': mention, 't_': 'sim',
				'jw': scores['jw'], 
				'jac' : scores['jac'], 
				'lev' : scores['lev'], 
				'spacy' : scores['spacy'],})
	except KeyboardInterrupt:
		raise KeyboardInterrupt
	except:
		with open(logname, 'a') as f:
			f.write(f'{mention}, {candidate}\n')


datasets = ['train', 'testA', 'testB']
datasets = [f'../../data/aida/template/full_{name}.csv' for name in datasets]
mentionCands = set()
for dataset in datasets:
	df = pd.read_csv(dataset)
	mentionCands.update(df.Mention_label.values)
mentionCands = [ml.split('===') for ml in mentionCands]
mentionCands = [(entry[0], entry[1]) for entry in mentionCands if (entry[0], entry[1]) not in known_feature_dict]

try:
	with multiprocessing.Pool(10) as pool:
		[ _ for _ in tqdm.tqdm(pool.imap_unordered(stringsim, mentionCands), total = len(mentionCands))]
except KeyboardInterrupt:
	save()
	sys.exit()
save()