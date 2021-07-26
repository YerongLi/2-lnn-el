import json
import os
import pandas as pd
import multiprocessing
import tqdm
import itertools
import pickle

def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists) 

pre = os.getenv("HOME") + '/lnn-el/data/aida/template/'
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
datasets = [pre + dataset for dataset in datasets]

pre = os.getenv("HOME") + '/DCA/'
djson = ['aida-train.json', 'aida-A.json', 'aida-B.json']
djson = [pre + j for j in djson]
for predictionfile, datasetfile in zip(djson, datasets):
	predicions = json.load(open(predictionfile, 'r'))
	df = pd.read_csv(datasetfile)
	feature_idx = df.columns.get_loc('Features')
	ml_idx = df.columns.get_loc('Mention_label')
	blink_idx = df.columns.get_loc('blink')

	print(len(set(df.Question.values)))
	feature_idx = df.columns.get_loc('Features')
	def process_dca(chunk):
		left, right = chunk
		assert(left < right)
		df_ = df.iloc[left: right]
		doc = str(df_.iloc[0].Question)

		n, l, r = df_.shape[0], 0, 0
		dataa = []
		count = 0
		while r < n:
			while r < n - 1 and df_.iloc[l]['QuestionMention'] == df_.iloc[r + 1]['QuestionMention']:
				r+= 1
			batch = df_.iloc[l: r + 1]
			# chunks.append([l, r + 1])
			# mention = batch.iloc[0].Mention
			# assert(mention == predicions[doc][count]['mention'])
			
			dt = batch.values.tolist()
			for i in range(len(dt)):
				cand = dt[i][ml_idx].split('===')[1].replace(' ', '_') 
				fv = eval(dt[i][feature_idx])
				if cand in predicions[doc][count]['pred']:
					fv[10] = float(predicions[doc][count]['pred'][cand]) + 0.5
					dt[i][blink_idx] = 1
					dt[i][feature_idx] = str(fv)
			l = r + 1
			r = l
			count+= 1
			dataa.extend(dt)
		return dataa
	recalculate = True
	if recalculate:
		chunks = []
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
		with open(predictionfile.split('/')[-1].split('.')[0] + '.pkl', 'rb') as fp:
			chunks = pickle.load(fp)
	with multiprocessing.Pool(40) as pool:
		data = list(tqdm.tqdm(pool.map(process_dca, chunks), total = len(chunks)))
	data = list(flatten(data))
	new_df = pd.DataFrame(data, columns=['left','Mention_label','Features','Label','Mention','QuestionMention','db','blink','right', 'Question'])
	# print(len(data))
	new_df.to_csv(datasetfile + '.2', index=False)
