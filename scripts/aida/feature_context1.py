import spacy
import py_stringmatching as sm
import pandas as pd
import pymongo
import multiprocessing
import tqdm
import sys
import os
import concurrent.futures
import json
import string
from nltk.corpus import stopwords


logname = os.path.basename(__file__) + '.log'

if os.path.exists(logname):
  os.remove(logname)

manager = multiprocessing.Manager()
stopwordset = manager.dict({word : 0 for word in stopwords.words('english')})
# # 建立mongodb连接
client = pymongo.MongoClient(host='localhost', port=27017)
ent_desc = manager.dict(json.load(open(os.getenv("HOME")+'/data/ent2desc.json', 'r')))

# # 连接stock数据库，注意只有往数据库中插入了数据，数据库才会自动创建
db = client.dbpedia

# # 创建一个daily集合，类似于MySQL中“表”的概念
# group = db['group']

features = db.features
document = db.document
document_dict = dict()
document_list = list(document.find())
with open('key.txt', 'w') as f:
	for entry in document_list:
		s = [k.lower() for k in entry['key1']]
		# print("str(set(entry['mention']))", entry['mention'])
		i =  entry['_id']
		document_dict[i] = list(set(s))
		# document_dict[entry['_id']] = str(set(entry['mention']))
del document_list
document_dict = manager.dict(document_dict)
feature_list = list(features.find({'t_': 'c1'}))

feature_dict = manager.list()
spacy_nlp = spacy.load("en_core_web_lg")
class Features:
	jw = sm.similarity_measure.jaro_winkler.JaroWinkler()
	lev = sm.similarity_measure.levenshtein.Levenshtein()
	smith_waterman = sm.similarity_measure.smith_waterman.SmithWaterman()
	partial_ratio = sm.similarity_measure.partial_ratio.PartialRatio()
	jac = sm.similarity_measure.jaccard.Jaccard()

known_feature_set = set()
for entry in  tqdm.tqdm(feature_list):
	known_feature_set.add((entry['d'],entry['men'], entry['cand']))
# feature_list = feature_list
def save():
	print('Saving the dicitonary to database')
	if feature_dict:
		features.insert_many(list(feature_dict))
	print(f'Saved {len(feature_dict)} Records.')
# ffcheck = open('British.txt', 'w')
def mentionOverlap(docEntity):
	global feature_dict
	docEntity = docEntity.split('===')
	doc, ent, label = docEntity[0], docEntity[1], docEntity[2]
	# print(doc, mention, cand)
	# print('doc, mention, cand')
	lbl = label.replace(' ', '_')
	try:
		context = document_dict[doc]
		description = ent_desc[lbl] if label in ent_desc else []
		description = ' '.join([word for word in description if word not in string.punctuation and word not in  stopwordset]).lower()
		if (doc, ent, label) not in known_feature_set:
			'''
			description, description of candidate entity
			context, the context keyword list of the mention
			ent, the mention
			label, the candidate entity
			'''	
			# simple exists overlap
			exists_score = 0
			if description is not None:
				if description == '':
					pass
				for item in context:
					if item == ent.lower():
						continue
					item = ' '.join([word for word in item.split() if word not in stopwords.words('english')])
					
					des = label.lower()+' ' + description
					score = Features.partial_ratio.get_sim_score(item, des) if item and des  else 0.0

					if score > 0.3 and exists_score < score:                    
						exists_score =score
			# if ent == 'British':
				# ffcheck.write((doc, ent, label, description, docEntity, exists_score, 'n'))
			feature_dict.append({'t_': 'c1', 'd': doc, 'men': ent, 'cand': label, 's_': exists_score})
	except KeyboardInterrupt:
		raise KeyboardInterrupt
	except:
		import traceback
		# traceback.print_exc()
		with open(logname, 'a') as f:
			f.write(f'{doc}, {label}\n')

datasets = ['train', 'testA', 'testB']
datasets = [f'../../data/aida/template/full_{name}.csv' for name in datasets]

docCands = set()
for dataset in datasets:
	df = pd.read_csv(dataset)
	df['DocCand'] = df.apply(lambda r : f"{r.Doc}==={r.Mention_label.split('===')[0]}==={r.Mention_label.split('===')[1]}", axis=1)
	docCands.update(df.DocCand.values)
try:
	with multiprocessing.Pool(40) as pool:
		[ _ for _ in tqdm.tqdm(pool.imap_unordered(mentionOverlap, docCands), total = len(docCands))]
except KeyboardInterrupt:
	save()
	# ffcheck.close()
	sys.exit()
save()
# ffcheck.close()
# mentionCands = [ml.split(';') for ml in mentionCands]
