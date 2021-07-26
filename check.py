import pandas as pd
import time
TYPE='Organization'
datasets = [f'data/aida/blink_bert_box/{TYPE}_train.csv', 
f'data/aida/blink_bert_box/{TYPE}_testA.csv',
f'data/aida/blink_bert_box/{TYPE}_testB.csv',
]
for dataset in datasets:
	df = pd.read_csv(dataset)
	df = df[df.Label == 1]
	print(len(set(df.Question.values.tolist())))
	print(df.shape[0])
# df = pd.read_csv('df.csv')

# print('Appending benchmark', end='\r')
# time.sleep(2)
# print('Extended benchmark ')
