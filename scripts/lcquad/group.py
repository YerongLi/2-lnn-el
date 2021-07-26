import pandas as pd
from tasks import filter_category
import xmltodict
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET

datasets = ['train', 'valid', 'test']
parser = argparse.ArgumentParser(description='Group all types')

parser.add_argument('type', type=str, nargs='?', help='', default = None)
args = parser.parse_args()
if not args.type:
    print('Input a type')

category = args.type              # picked
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
# category = 'Sport'
# category = 'BiologicalLivingObject'
# category = 'FictionalCharacter'



for name in datasets:
    fname = f'../data/lcquad/blink_bert_box/full_{name}_gold.csv'
    f = open(fname, "r")
    f.readline()
    df = pd.read_csv(fname)
    df.sort_values(by=['QuestionMention'])
    n, l, r = df.shape[0], 0, 0
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        lines = ''.join([f.readline() for i in range(r - l + 1)])
        filter_category.apply_async((df.iloc[l: r + 1].to_json(), lines, category, name))
        l = r + 1
        r = l
    f.close()
