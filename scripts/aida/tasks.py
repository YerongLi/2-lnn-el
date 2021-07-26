import pymongo, re, time, warnings
import json
import tqdm
import sys
from celery import Celery
from urllib.error import HTTPError
import traceback, os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
import ast
from nltk.wsd import lesk
import spacy
import pytextrank
from nltk.corpus import stopwords
spacy_nlp = spacy.load("en_core_web_lg")
spacy_nlp.add_pipe("textrank")
import py_stringmatching as sm
import numpy as np
import sklearn
class Features:
	jw = sm.similarity_measure.jaro_winkler.JaroWinkler()
	lev = sm.similarity_measure.levenshtein.Levenshtein()
	smith_waterman = sm.similarity_measure.smith_waterman.SmithWaterman()
	partial_ratio = sm.similarity_measure.partial_ratio.PartialRatio()
	jac = sm.similarity_measure.jaccard.Jaccard()
# 设置BROKER
BROKER_URL = 'mongodb://127.0.0.1:27017/celery'

# 新建celery任务
app = Celery('my_task', broker=BROKER_URL)


# # 建立mongodb连接
client = pymongo.MongoClient(host='localhost', port=27017)

# # 连接stock数据库，注意只有往数据库中插入了数据，数据库才会自动创建
db = client.dbpedia

# # 创建一个daily集合，类似于MySQL中“表”的概念
group = db['group']

features = db['features']

redirect = db['redirect']

@app.task
def redirect_link(original):
	if redirect.find_one({'_id': original}) is not None: return
	r = requests.get(f'http://en.wikipedia.org/wiki/{original}')
	# print(r.content, 'content')
	tmp = str(r.content).replace('<link rel="canonical" href="', 'r@ndom}-=||').split('r@ndom}-=||')[-1]
	idx = tmp.find('"/>')
	real_link = tmp[:idx]
	real = real_link.split('/')[-1]
	if original != real:
		redirect.insert_one({'_id': original, 'real':real})


@app.task
def lookup(query):
	link = f'https://lookup.dbpedia.org/api/search/KeywordSearch?QueryString=%22{query}%22&MaxHits=800'
	filename = f'../data/aida/candidates/{query}.xml'
	try:
		# if not os.path.isfile(filename):
		if True:
			r = requests.get(link)
			open(filename, 'wb').write(r.content)
	except Exception as e:
		traceback.print_exc()

@app.task
def save_type(entity):
	link =f'https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=PREFIX+owl%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2002%2F07%2Fowl%23%3E%0D%0APREFIX+xsd%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+rdf%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0D%0APREFIX+foaf%3A+%3Chttp%3A%2F%2Fxmlns.com%2Ffoaf%2F0.1%2F%3E%0D%0APREFIX+dc%3A+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Felements%2F1.1%2F%3E%0D%0APREFIX+%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F%3E%0D%0APREFIX+dbpedia2%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0D%0APREFIX+dbpedia%3A+%3Chttp%3A%2F%2Fdbpedia.org%2F%3E%0D%0APREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0ASELECT+*+WHERE+%7B%0D%0A%7B+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F{entity}%3E+a+%3Ftype%7D%0D%0A%7D%0D%0A&output=json'
	try:
		os.stat('type')
	except:
		os.mkdir('type')
	filename = f'type/{entity}.json'
	if os.path.exists(filename):
		pass
	try:
		with requests.get(link) as url:
			# pass
			data = json.loads(url.text)
			json.dump(data, open(filename, 'w'))
	except:
		with open('log.txt', 'a') as f:
			traceback.print_exc()
			f.write(traceback.format_exc())
			f.write(f"CANNOT type file : '{entity}',\n")
			f.write(link)

@app.task
def filter_category(df, lines, category, dataset):
	print(category)
	df = pd.read_json(df)

	ground_truth = df[df.Label.eq(1)]
	if ground_truth.shape[0] == 0:
		return
	

	ground_truth = ground_truth.iloc[0]
	if group.find_one({'QM': ground_truth.QuestionMention, 'set' : dataset, 'type': category}) is not None: return
	mention_candidate =str(ground_truth.Mention_label).split(';')
	mention, candidate = mention_candidate[0], mention_candidate[1]
	found = False
	try:
		if os.path.exists(f'../data/lcquad/candidates/{mention}.xml'):
			try:
				with open(f'../data/lcquad/candidates/{mention}.xml') as fd:
					tree = ET.fromstring(fd.read())
					candidates = tree.findall('Result')
			except:
				with open('log.txt', 'a') as f:
					traceback.print_exc()
					f.write(traceback.format_exc())
					f.write(f"CANNOT open file : '{mention}',\n")
				return
			for c in candidates:
				na = c.find('Label').text
				if na == candidate:
					classes = c.find('Classes').findall('Class')
					for cl in classes:
						s = ''.join([i for i in cl.find('URI').text.split('/')[-1] if i.isalpha()])
						if s == category:
							group.insert_one({'QM': ground_truth.QuestionMention, 'set' : dataset, 'type': category,'row' : lines})
							found = True
							break
					break


		else:
			with open('log.txt', 'a') as f:
				traceback.print_exc()
				f.write(traceback.format_exc())
				f.write(f"NO such file : '{mention}'xml,\n")
	except:
		with open('log.txt', 'a') as f:
			traceback.print_exc()
			f.write(traceback.format_exc())
	# found = False
	if found : return
	if group.find_one({'QM': ground_truth.QuestionMention, 'set' : dataset, 'type': category}) is not None: return
	try:
		entity = candidate.replace(' ', "_")
		if os.path.exists(f'type/{entity}.json'):
			try:
				with open(f'type/{entity}.json') as f:
					data = json.loads(f.read())

					classes = set([''.join([c for c in i['type']['value'].split('/')[-1] if c.isalpha()])
					 for i in data['results']['bindings']])
					for cl in classes:
						if cl == category:
							group.insert_one({'QM': ground_truth.QuestionMention, 'set' : dataset, 'type': category,'row' : lines})
							break
			except:
				with open('log.txt', 'a') as f:
					traceback.print_exc()
					f.write(traceback.format_exc())
					f.write(f"KANNOT open file : '{entity}',\n")
		else:
			with open('log.txt', 'a') as f:
				traceback.print_exc()
				f.write(traceback.format_exc())
				f.write(f"NO such file : '{entity}'json,\n")
	except:
			with open('log.txt', 'a') as f:
				traceback.print_exc()
				f.write(traceback.format_exc())


@app.task
def append_feature(df, dataset):
	df = pd.read_json(df)
	if df.shape[0] == 0:
		return
	mention = df.iloc[0]['Mention']
	if os.path.exists(f'../data/lcquad/candidates/{mention}.xml'):
		try:
			with open(f'../data/lcquad/candidates/{mention}.xml') as fd:
				tree = ET.fromstring(fd.read())
				candidates = tree.findall('Result')
		except:
			with open('log.txt', 'a') as f:
				traceback.print_exc()
				f.write(traceback.format_exc())
				f.write(f"CANNOT open file : '{mention}',\n")
			return
	else:
		print(os.path.exists(f'../data/lcquad/candidates/{mention}.xml'))
		print(f"Skipped {mention}.xml")
		return # DEBUG
	for i in range(df.shape[0]):
		row = df.iloc[i]
		l = f"{row['Question']},{row['Mention_label']},{row['Features']},{row['Label']},{row['Mention']},{row['QuestionMention']},{row['db']},{row['blink']}"
		mention_candidate =str(row.Mention_label).split(';')
		_, candidate = mention_candidate[0], mention_candidate[1]
		sentence0 = lesk(row['Question'].split(), mention, 'n')
		if sentence0 is not None:
			sentence0 = sentence0.definition()
		newfeature = ast.literal_eval(row['Features'])
		sense_sim = 0.0
		try:
			if features.find_one({'QM': row.QuestionMention, 'cand' :candidate, 'set' : dataset}) is None:
				found = False
				for c in candidates:
					na = c.find('Label').text
					if na == candidate:
						sentence1 = c.find('Description').text
						if sentence0 is not None and sentence1 is not None:
							sentence_embeddings = model.encode([sentence0, sentence1])
							sense_sim = cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0]
							# print(cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]])[0][0])		
				newfeature.append(sense_sim)
				l = f"{row['Question']},{row['Mention_label']},{str(newfeature)},{row['Label']},{row['Mention']},{row['QuestionMention']},{row['db']},{row['blink']}"
				features.insert_one({'QM': row.QuestionMention, 'cand': candidate, 'set': dataset, 'row' : l})
				# print(candidates)
			# else:
			# 	pass
			# 	# print(f'found record')
		except:
			print(mention, row['Question'])
			with open('log.txt', 'a') as f:
				traceback.print_exc()
				f.write(traceback.format_exc())

	# sentence_embeddings = model.encode(sentences)

datasets = ['aida_train.txt_data.csv', 'temp_aida_dev_data.csv', 'temp_aida_test_data.csv']
d = {}

for dataset in datasets:
    gtname = '_'.join(dataset.split('_')[:-1]) + '_groundtruth.csv'
    gt = pd.read_csv(gtname)
    for i in tqdm.tqdm(range(gt.shape[0])):
        k = gt.iloc[i].Doc + '===' + gt.iloc[i].Sentence
        if k in d: continue
        else:
            d[k] = gt.iloc[i].Environment

@app.task
def feature_context1(df, dataset):
	'''
		Overlapping score named as c1 
	'''
	df = pd.read_json(df)
	# print(d['DOCSTART_1_EU===EU rejects German call to boycott British lamb .'])
	# mention, candidate, context, question, doc
	# print(d)

	context = d[df.iloc[0].Question]
	question = df.iloc[0].Question
	doc = df.iloc[0].Question.split('===')[0]
	# add PyTextRank to the spaCy pipeline
	context = spacy_nlp(context)

	# examine the top-ranked phrases in the document
	context = [phrase.text for phrase in context._.phrases]
	s = df.Mention_label.values
	s = set([it.split(';')[-1] for it in s])
	for mention in set(df.Mention.values):
		if os.path.exists(f'../../data/candidates/{mention}.xml'):
			try:
				with open(f'../../data/candidates/{mention}.xml') as fd:
					tree = ET.fromstring(fd.read())
					candidates = tree.findall('Result')
			except:
				with open('log.txt', 'a') as f:
					traceback.print_exc()
					f.write(traceback.format_exc())
					f.write(f"CANNOT open file : '{mention}',\n")
					return
		else:
			print(os.path.exists(f'../../data/lcquad/candidates/{mention}.xml'))
			print(f"Skipped {mention}.xml")
		count = 0
		for c in candidates:
			na = c.find('Label').text
			if na in s:
				if features.find_one({'Q': question, 'cand' :na, 'men': mention, 
			't_': 'c1', 'set' : dataset, 'doc': doc}) is not None:
					continue
				# print(na)
				description = c.find('Description').text
				score = overlap_score(description, context, mention, na)
				features.insert_one({'Q': question, 'cand' :na, 'men': mention, 
					't_': 'c1', 's_': score, 'set' : dataset, 'doc': doc})
				count+= 1
				if count == len(s):
					break


def overlap_score(description, context, ent, label):
	'''
	description, description of candidate entity
	context, the context keyword list of the mention
	ent, the mention
	label, the candidate entity
	'''	
	# simple exists overlap
	exists_count = 0
	exists_score = 0
	if description is not None:
		if description == '':
			pass
			# description = str(self.get_abstract(label))
			#print('getting new description...'+description)
		for item in context:
			if item == ent.lower():
				continue
			tmp = []
			tmp.append(description)
			item = ' '.join([word for word in item.split() if word not in stopwords.words('english')])
			
			description = label+' '+description
			scores = _get_sim_scores(item, description.lower())
			# if label == "British Columbia" or label == "Comox, British Columbia":
			#     print("{} {} {}".format(item, label,scores))
			#if (scores['jw'] > self.threshold or scores['jacc'] > self.threshold or scores['lev'] > self.threshold or scores['spacy'] > self.threshold):
			#if scores['in'] > 0.0:
			#    exists_count += scores['in']
			if scores['pr'] > 0.3 and exists_score < scores['pr']:                    
				exists_score = scores['pr']
				#exists_count += 1
			#elif scores['smith_waterman'] > 3.0:
			#    exists_count += scores['smith_waterman']/len(item)
	# if label == "British Columbia" or label == "Comox, British Columbia":
	#     print(label, exists_score)
	return exists_score#/exists_count if exists_count > 0  else exists_score

def _get_sim_scores(str1, str2):
		str1 = str1.lower()
		str2 = str2.lower()
		scores = {}
		# if str1 == str2:
		# 	scores['exact'] = 1.0
		# else:
		# 	scores['exact'] = 0.0
		# if str2.count(str1) >= 1:
		# 	scores['in'] = 1.0
		# else:
		# 	scores['in'] = 0.0
		# scores['jw']=self.jw.get_sim_score(str1,str2)
		# scores['jacc'] = self.jacc.get_sim_score(str1.split(' '),str2.split(' '))
		# scores['lev'] = Features.lev.get_sim_score(str1,str2)
		# scores['spacy'] = spacy_nlp(str1).similarity(spacy_nlp(str2))
		# scores['smith_waterman'] = Features.smith_waterman.get_raw_score(str1, str2) if str1 and str2  else 0.0
		scores['pr'] = Features.partial_ratio.get_sim_score(str1, str2) if str1 and str2  else 0.0
		#scores['bss'] = self.nss.get_bert_sim(str1,str2, sim_type='cosine')
		return scores

@app.task
def stringsim(mention, candidate):
	if features.find_one({'cand' :candidate, 'men': mention, 
	't_': 'sim'}) is None:
		scores = dict()
		scores['jw']= Features.jw.get_sim_score(mention,candidate)
		scores['jac'] = Features.jac.get_sim_score(mention.split(' '),candidate.split(' '))
		scores['lev'] = Features.lev.get_sim_score(mention,candidate)
		scores['spacy'] = spacy_nlp(mention).similarity(spacy_nlp(candidate))
		# scores['smith_waterman'] = Features.smith_waterman.get_raw_score(mention, candidate) if mention and candidate  else 0.0
		# scores['pr'] = Features.partial_ratio.get_sim_score(mention, candidate) if mention and candidate  else 0.0
		features.insert_one({'cand' :candidate, 'men': mention, 't_': 'sim',
			'jw': scores['jw'], 
			'jac' : scores['jac'], 
			'lev' : scores['lev'], 
			'spacy' : scores['spacy']})

@app.task
def feature_count(df):
	df = pd.read_json(df)
	mention = df.iloc[0].Mention
	question = df.iloc[0].Question
	if features.find_one({'men': mention, 'Q': question,
		't_': 'ref'}) is None:
		if os.path.exists(f'../../data/candidates/{mention}.xml'):
			try:
				with open(f'../../data/candidates/{mention}.xml') as fd:
					tree = ET.fromstring(fd.read())
					candidates = tree.findall('Result')
			except:
				with open('log.txt', 'a') as f:
					traceback.print_exc()
					f.write(traceback.format_exc())
					f.write(f"CANNOT open file : '{mention}',\n")
					return
		else:
			print(os.path.exists(f'../../data/lcquad/candidates/{mention}.xml'))
			print(f"Skipped {mention}.xml")
		s = set([it.split(';')[-1] for it in df['Mention_label'].values])
		result = dict()
		count = 0
		for c in candidates:
			na = c.find('Label').text
			ref_count = c.find('Refcount').text
			if ref_count is not None:
				ref_count = int(ref_count)
			else:
				ref_count = 0
			if na in s:
				result[na] = ref_count
				count+= 1
				if count == len(s):
					break

		s = [it.split(';')[-1] for it in df['Mention_label'].values]
		ref_scores = [result[e] + 1 for e in s]
		if len(ref_scores) > 1:
			normalized_ref_scores = sklearn.preprocessing.minmax_scale(list(np.log(ref_scores)), feature_range=(0.1, 1))#self.normalize(ref_scores)
		else:
			normalized_ref_scores = [1.0]
		for i, cand in enumerate(s):
			result[cand] = normalized_ref_scores[i]
		features.insert_one({'men': mention, 'Q': question, 't_': 'ref', 'count' : str(result)})
