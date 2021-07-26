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
from sentence_transformers import SentenceTransformer
import ast
from nltk.wsd import lesk
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

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


@app.task
def lookup(query):
	link = f'https://lookup.dbpedia.org/api/search/KeywordSearch?QueryString=%22{query}%22&MaxHits=800'
	filename = f'../data/lcquad/candidates/{query}.xml'
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


@app.task
def modified_lev(df, name):
	print
