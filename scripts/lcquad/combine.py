import pandas as pd

# categories = [
#     'Person',
#     'Species',
#     'CreativeWork',
#     'Organization', 
#     'Event',
#     'TelevisionShow',
#     'Location',
#     'Team',
#     'Vehicle',
#     'Food',
#     'Software',
#     ]

categories = [
    'Person',
    # 'Species',
    # 'CreativeWork',
    # 'Organization', 
    'Event',
    # 'TelevisionShow',
    # 'Location',
    # 'Team',
    # 'Vehicle',
    # 'Food',
    # 'Software',
    ]
combined_name = 'combined'
# combined_name = ''.join(categories)
datasets = ['train', 'valid', 'test']

for name in datasets:
    li = []
    for category in categories:
        fname = f'../data/lcquad/blink_bert_box/{category}_{name}.csv'
        li.append(pd.read_csv(fname))
    df = pd.concat(li, axis=0, ignore_index=True)
    df.drop(['row_num','Unnamed: 0'], axis=1, errors='ignore', inplace=True)

    df.drop_duplicates(inplace = True)
    df.to_csv(f'../data/lcquad/blink_bert_box/{combined_name}_{name}.csv', index = False)
    del df

li = []
for category in categories:

    li.append(pd.read_csv(f'../data/lcquad/{category}_lcquad_gt_5000.csv'))

df = pd.concat(li, axis=0, ignore_index=True)
df.drop(['Unnamed: 0'], axis=1, errors='ignore', inplace=True)
df.drop_duplicates(inplace = True)
df.to_csv(f'../data/lcquad/{combined_name}_lcquad_gt_5000.csv', index = False)