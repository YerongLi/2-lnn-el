import argparse
parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")

parser.add_argument("--type", type=str, default='full')

args = parser.parse_args()
# TYPE = 'Person'
TYPE = args.type
assert(TYPE != 'full')
import json
import os
import pandas as pd
import multiprocessing
import tqdm
import itertools
import pickle
import pymongo


def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists) 

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
		document_dict[i] = [k[1] for k in entry['mention'] if TYPE == 'full' or (k[1] in tjson and tjson[k[1]] == mtype2id[TYPE])]
		# document_dict[entry['_id']] = str(set(entry['mention']))
pre = os.getenv("HOME") + '/lnn-el/data/aida/blink_bert_box//'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + dataset for dataset in datasets]

pre = os.getenv("HOME") + '/DCA/'
djson = ['aida-train.json', 'aida-A.json', 'aida-B.json']
djson = [pre + j for j in djson]
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
		n, l, r = df_.shape[0], 0, 0
		data = []
		# print(df_.shape, 'shape')
		while r < n:
			while r < n - 1 and df_.iloc[l]['QuestionMention'] == df_.iloc[r + 1]['QuestionMention']:
				r+= 1
			batch = df_.iloc[l : r + 1]
			try:
				ground_truth = batch[batch.Label.eq(1)].iloc[0].Mention_label.split('===')[1].replace(' ', '_')
			except:
				import sys, traceback
				traceback.print_exc()
				print(batch.iloc[0].Question, left, right, l, r)
				print(batch.iloc[0].Mention)
				sys.exit()
			if TYPE == 'full' or (ground_truth in tjson and tjson[ground_truth] == mtype2id[TYPE]):
				data.extend(batch.values.tolist()) 
			l = r + 1
			r = l
		if len(data) == 0: return None
		return data

	feature_idx = df.columns.get_loc('Features')

	recalculate = False
	if recalculate:
		n, l, r = df.shape[0], 0, 0
		# print(len(set(df.QuestionMention.values)))
		while r < n:
			while r < n - 1 and df.iloc[l]['Question'] == df.iloc[r + 1]['Question']:
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
	# chunks = [chunks[0]]
	print('Begin mapping')
	with multiprocessing.Pool(40) as pool:
		data = list(tqdm.tqdm(pool.map(process, chunks), total = len(chunks)))
	filename = os.getenv("HOME") + '/lnn-el/data/aida/' + TYPE + '_aida_gt_5000.csv'
	data = [d for d in data if d is not None]
	data = list(flatten(data))
	
	new_df = pd.DataFrame(data, columns=['left','Mention_label','Features','Label','Mention','QuestionMention','db','blink','right', 'Question'])
	new_df.to_csv(datasetfile.replace('full', TYPE), index=False)


