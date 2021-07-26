import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
def key(question):
	print(question)
	sentence = Sentence(question)

	# load the NER tagger
	tagger = SequenceTagger.load('ner')

	# run NER over sentence
	tagger.predict(sentence)
	print(sentence)
	print('The following NER tags are found:')
	print([t for t in sentence.get_spans('ner')])

df = pd.read_csv(f'../data/lcquad/full_lcquad_gt_5000.csv')
df.Question.apply(key)
