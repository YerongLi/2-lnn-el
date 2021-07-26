import pandas as pd
from tqdm import tqdm
import ast
import os
import json
category = 'Person'
h = open(f'../data/lcquad/lcquad_gt_5000.csv', 'r')
f = open(f'../data/lcquad/{category}lcquad_gt_5000.csv', 'w')
f.write(h.readline())
h.close()

benchmark = pd.read_csv(f'../data/lcquad/lcquad_gt_5000.csv')
benchmark['Entities'] = benchmark['Entities'].astype(object)
idx = 0
for  _ , gold_row in tqdm(benchmark.iterrows()):
    gold_entities = ast.literal_eval(gold_row['Entities'])
    valid_entity = set()
    # line = h.readline()
    for entity in gold_entities:
        filename = f'type/{entity}.json'
        if not os.path.exists(filename):
            continue
        typejson = json.load(open(filename, 'r'))
        typelist = typejson['results']['bindings']
        typelist = set([item['type']['value'].split('/')[-1] for item in typelist])
        if category in typelist:
            valid_entity.add(entity)
    # print(line)
    if valid_entity:
        line = f"\"{idx},{gold_row['Question']}\",\"{gold_row['Classes']}\",\"{valid_entity}\"\n"
        f.write(line)
        idx+= 1
    # line = h.readline()
f.close()
