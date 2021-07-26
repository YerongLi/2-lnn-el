

# Get the dataset
import xml.etree.ElementTree as ET
import urllib.request
import xmltodict
url = 'https://lookup.dbpedia.org/api/search/KeywordSearch?QueryString=%22Taiko%22&MaxHits=200'

# url = 'https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/fe9e8b7d-61ea-409d-84aa-3ebd79a046b5.xml'
response = urllib.request.urlopen(url).read()
tree = ET.fromstring(response)

# print(str(tree))
candidates = tree.findall('Result')
# with open("xml_file.xml") as xml_file:
data_dict = xmltodict.parse(tree)
print('data_dict', data_dict)
print(candidates[0])
# ind = 183
# print(candidates[ind].find('Label').text)
# print(candidates[ind].find('Description').text) 
      

# # for compTitle in tree.findall('.//{urn:hl7-org:v3}title'):
# #       print(compTitle.text)

# import pandas as pd
# import urllib.request, json
# from tqdm import tqdm
# # cnt = 0
# # name  = 'train'
# # name  = 'valid'
# name  = 'test'
# csvfile = pd.read_csv(f'../data/lcquad/blink_bert_box/{name}_gold.csv')
# outdf = pd.DataFrame(columns= csvfile.columns)

# for _, row in tqdm(csvfile.iterrows()):
#     # print(row)
#     # cnt+= 1
#     # if cnt > 10: break
#     # continue
#     pair = row['Mention_label']
#     # print(row.keys())
#     mention_candidate =str(pair).split(';')
#     # mention = mention_candidate[0]
#     candidate = mention_candidate[1].replace(' ', '_')
#     # print(candidate)

#     link =f'https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=PREFIX+owl%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2002%2F07%2Fowl%23%3E%0D%0APREFIX+xsd%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+rdf%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0D%0APREFIX+foaf%3A+%3Chttp%3A%2F%2Fxmlns.com%2Ffoaf%2F0.1%2F%3E%0D%0APREFIX+dc%3A+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Felements%2F1.1%2F%3E%0D%0APREFIX+%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F%3E%0D%0APREFIX+dbpedia2%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0D%0APREFIX+dbpedia%3A+%3Chttp%3A%2F%2Fdbpedia.org%2F%3E%0D%0APREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0ASELECT+*+WHERE+%7B%0D%0A%7B+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F{candidate}%3E+a+%3Ftype%7D%0D%0A%7D%0D%0A&output=json'
#     # print(link)
#     try:
#         with urllib.request.urlopen(link) as url:
#             data = json.loads(url.read().decode())
#             for item in data['results']['bindings']:
#                 if item['type']['value'][-7:] == '/Person':
#                     outdf.loc[len(outdf.index)] = row
                    
#                     print(candidate)
#                     if 0 == len(outdf.index) % 10:
#                         outdf.to_csv(f'new_{name}.csv')
#                     break

#     except:
#         import traceback
#         traceback.print_exc()
#         # continue
# outdf.to_csv(f'new_{name}.csv')


# # # print(candidate[1].replace(' ', '_'))
# # link= "https://dbpedia.org/data/Salt_Lake_City.jsod"
# # import urllib.request, json 
# # with urllib.request.urlopen(link) as url:
# #     data = json.loads(url.read().decode())
# #     print(data)