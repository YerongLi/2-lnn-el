TYPE = 'full'
import json
import os
import pandas as pd
import multiprocessing
import tqdm
import itertools
import pickle
import pymongo

client = pymongo.MongoClient(host='localhost', port=27017)
db = client.dbpedia

document = db.document

document_dict = dict()
document_list = list(document.find())
tjson = json.load(open(os.getenv('HOME') + '/data/entityType.json', 'r'))

mtype2id = {'Person':0, 'Organization':1, 'Location':2, 'UNK':3, 'full' : 6}

with open('key.txt', 'w') as f:
	for entry in document_list:
		# print("str(set(entry['mention']))", entry['mention'])
		i =  entry['_id']
		# f.write(i + '\n')
		document_dict[i] = [k for k in entry['gold'] if TYPE == 'full' or (k in tjson and tjson[k] == mtype2id[TYPE])]
		# document_dict[entry['_id']] = str(set(entry['mention']))
pre = os.getenv("HOME") + '/lnn-el/data/aida/template/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + dataset for dataset in datasets]

pre = os.getenv("HOME") + '/DCA/'
djson = ['aida-train.json', 'aida-A.json', 'aida-B.json']
djson = [pre + j for j in djson]
all_data = []
for predictionfile, datasetfile in zip(djson, datasets):
	df = pd.read_csv(datasetfile)
	feature_idx = df.columns.get_loc('Features')
	ml_idx = df.columns.get_loc('Mention_label')
	blink_idx = df.columns.get_loc('blink')

	# print(len(set(df.Doc.values)))
	def process(chunk):
		left, right = chunk
		assert(left < right)
		df_ = df.iloc[left: right]
		doc = str(df_.iloc[0].Doc)
		return [doc, document_dict[doc]]
	feature_idx = df.columns.get_loc('Features')

	recalculate = False
	if recalculate:
		n, l, r = df.shape[0], 0, 0
		# print(len(set(df.QuestionMention.values)))
		while r < n:
			while r < n - 1 and df.iloc[l]['Doc'] == df.iloc[r + 1]['Doc']:
				r+= 1
			chunks.append([l, r + 1])
			# lines = ''.join([f.readline() for i in range(r - l + 1)])
			# feature_count.apply_async([df.iloc[l: r + 1].to_json()])
			# feature_count(df.iloc[l: r + 1].to_json())
			l = r + 1
			r = l
		with open(predictionfile.split('/')[-1].split('.')[0] + '.pkl', 'wb') as fp:
			pickle.dump(chunks, fp)
	else:
		with open(os.getenv("HOME") + '/lnn-el/scripts/aida/fill/' + predictionfile.split('/')[-1].split('.')[0] + '.pkl', 'rb') as fp:
			chunks = pickle.load(fp)
	with multiprocessing.Pool(40) as pool:
		data = list(tqdm.tqdm(pool.map(process, chunks), total = len(chunks)))
	all_data.extend(data)
filename = os.getenv("HOME") + '/lnn-el/data/aida/' + TYPE + '_aida_gt_5000.csv'
new_df = pd.DataFrame(all_data, columns=['Question','gold'])
new_df.to_csv(filename, index=False)


