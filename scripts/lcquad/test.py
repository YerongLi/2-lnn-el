import pandas as pd
# import ast
from flair.data import Sentence
from flair.models import SequenceTagger
df = pd.read_csv(f'../data/lcquad/full_lcquad_gt_5000.csv')
for i in range(df.shape[0]):
	question = df.iloc[i]['Question']

	question = 'List the scientists whose doctoral advisor is Ernest Rutherford and are known for Manhattan Project?'
	sentence = Sentence(question)

	# load the NER tagger
	tagger = SequenceTagger.load('ner')

	# run NER over sentence
	tagger.predict(sentence)
	print(sentence)
	print('The following NER tags are found:')
	print(type(sentence.get_spans('ner')[0]))
	break
	# # iterate over entities and print
	# for entity in sentence.get_spans('ner'):
	# 	print(entity)
	# break

