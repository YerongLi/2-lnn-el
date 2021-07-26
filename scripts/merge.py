import pandas as pd
import os
import tqdm
datasets = ['full_train.csv', 'full_testA.csv', 'full_testB.csv']
for dataset in datasets:
	filename = os.getenv('HOME') + '/lnn-el/data/aida/blink_bert_box/' + dataset
	df = pd.read_csv(filename)
	for i in tqdm.tqdm(range(df.shape[0])):
		# print(df.iloc[i].QuestionMention)
		df.iloc[i, df.columns.get_loc('QuestionMention')] = \
		str(df.iloc[i].Question) + ' ' + str(df.iloc[i].left) + str(df.iloc[i].right)
		# print(df.iloc[i].QuestionMention)
	# df.to_csv(filename, index=False)