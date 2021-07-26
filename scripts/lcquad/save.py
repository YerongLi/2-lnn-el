import pymongo
import pandas as pd
from tqdm import tqdm
import argparse

# # 建立mongodb连接
client = pymongo.MongoClient(host='localhost', port=27017)

# # 连接stock数据库，注意只有往数据库中插入了数据，数据库才会自动创建
db = client.dbpedia

parser = argparse.ArgumentParser(description='Group all types')

parser.add_argument('type', type=str, nargs='?', help='', default = None)
args = parser.parse_args()
if not args.type:
    print('Input a type')
category = args.type              # picked

# category = 'Person'               # picked
# category = 'Species'              # picked
# category = 'Book'                 
# category = 'Company'                  
# category = 'Organization'                  # picked
# category = 'Scientist'
# category = 'Event'                    # picked
# category = 'Actor' 
# category = 'TelevisionShow'           # picked
# category = 'WrittenWork'            
# category = 'Politician'
# category = 'AmericanFootballPlayer'
# category = 'Location'                    # picked
# category = 'School'             
# category = 'Project'
# category = 'Film'            
# category = 'Team'                      #picked
# category = 'Award'

# category = 'Vehicle'                  # picked
# category = 'Food'                     # picked
# category = 'CreativeWork'             # picked
# category = 'PersonFunction'             
# category = 'Software'
# category = 'Language'             
# category = 'Currency'

group = db['group']

datasets = ['train', 'valid', 'test']
# datasets = ['test']

for name in datasets:
    h = open(f'../data/lcquad/blink_bert_box/full_{name}_gold.csv', 'r')
    f = open(f'../data/lcquad/blink_bert_box/{category}_{name}.csv', 'w')
    f.write(h.readline())
    h.close()
    for entry in tqdm(group.find({'set' : name, 'type' : category})):
        f.write(entry['row'])
    f.close()
out = open(f'../data/lcquad/{category}_lcquad_gt_5000.csv', 'w')
out.write('ID,Question,Classes,Entities\n')

for name in datasets:
    # out = open(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv', 'w')
    # out.write('ID,Question,Classes,Entities\n')

    df = pd.read_csv(f'../data/lcquad/blink_bert_box/{category}_{name}.csv')
    n, l, r = df.shape[0], 0, 0
    idx = 0
    while r < n:
        while r < n - 1 and df.iloc[l]['Question'] == df.iloc[r + 1]['Question']:
            r+= 1
        batch = df.iloc[l : r + 1]
        gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
        # gold_entities = set([pair.split(';')[-1].replace(' ', '_') for pair in gold_pairs if not "'" in pair])
        gold_entities = set([str(pair.split(';')[-1].replace(' ', '_')).replace("'", "\t").replace('"', "'") for pair in gold_pairs])
        gold_entities = str(gold_entities)

        if df.iloc[l].Question == 'Did John Byrne create Emma Frost?':
            print(gold_pairs)
            print(gold_entities) 
        line = f'{idx},\"{df.iloc[l].Question}\",[],\"{gold_entities}\"\n'
        out.write(line)
        idx+= 1
        l = r + 1
        r = l
    # out.close()
    # _ = pd.read_csv(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
    # print(f'[Tested] ../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
out.close()
_ = pd.read_csv(f'../data/lcquad/{category}_lcquad_gt_5000.csv')
print(f'[Tested] ../data/lcquad/{category}_lcquad_gt_5000.csv')
savefull = True
if savefull:
    full = 'full'
    out = open(f'../data/lcquad/{full}_lcquad_gt_5000.csv', 'w')
    out.write('ID,Question,Classes,Entities\n')

    for name in datasets:
        # out = open(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv', 'w')
        # out.write('ID,Question,Classes,Entities\n')

        df = pd.read_csv(f'../data/lcquad/blink_bert_box/{name}_gold.csv')
        n, l, r = df.shape[0], 0, 0
        idx = 0
        while r < n:
            while r < n - 1 and df.iloc[l]['Question'] == df.iloc[r + 1]['Question']:
                r+= 1
            batch = df.iloc[l : r + 1]
            gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
            # gold_entities = set([pair.split(';')[-1].replace(' ', '_') for pair in gold_pairs if not "'" in pair])
            gold_entities = set([str(pair.split(';')[-1].replace(' ', '_')).replace("'", "\t").replace('"', "'") for pair in gold_pairs])
            gold_entities = str(gold_entities)
            # exceptions = [pair.split(';')[-1].replace(' ', '_') for pair in gold_pairs if "'" in pair]
            # if (len(exceptions) > 0):
            #     print('ex',exceptions)
            #     print(gold_entities)

            if len(gold_entities) > 0:
                line = f'{idx},\"{df.iloc[l].Question}\",[],\"{gold_entities}\"\n'
                out.write(line)
                idx+= 1
            l = r + 1
            r = l
        # out.close()
        # _ = pd.read_csv(f'../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
        # print(f'[Tested] ../data/lcquad/{category}_{name}_lcquad_gt_5000.csv')
    out.close()
    _ = pd.read_csv(f'../data/lcquad/{full}_lcquad_gt_5000.csv')
    print(f'[Tested] ../data/lcquad/{full}_lcquad_gt_5000.csv')