import pandas as pd
df = pd.read_csv('temp.csv')
mention = 'Rugby Union'
groundtruth = 'Rugby Union'
mistake = 'University of Tasmania Rugby Union Club'

gt = df[df['Label'] == 1]
gt = gt[df['Mention'] == mention]
print(gt.iloc[0].Features, gt.iloc[0].blink)
# print(df[df['Mention_label'] == f'{mention}==={groundtruth}'].iloc[0].Features)
print()
print('mistakes')
print(df[df['Mention_label'] == f'{mention}==={mistake}'].iloc[0].Features,
df[df['Mention_label'] == f'{mention}==={mistake}'].iloc[0].blink)
print(gt.Mention_label)

df = pd.read_csv('full_train.csv.2')

gt = df[df['Label'] == 1]
blk = gt[gt.blink == 1]
print(gt.shape[0])
print(blk.shape[0])
