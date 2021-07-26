# Investaging the difference between the df2 subset and the training set
import pandas as pd
fname = f'../data/lcquad/blink_bert_box/train_gold.csv'
cname = f'../data/lcquad/blink_bert_box/combined_train.csv'
f = open(fname, "r")
f.readline()
df1a = pd.read_csv(fname)[['QuestionMention','Label','Mention_label']]
df2a = pd.read_csv(cname)[['QuestionMention','Label', 'Mention_label']]
df1a.drop_duplicates()
df2a.drop_duplicates()

print('df1')
print(df1a.shape[0])
df1 = df1a[df1a['Label'] == 1]

print(df1.shape[0])


print('df2')
print(df2a.shape[0])
df2 = df2a[df2a['Label'] == 1]
print(df2.shape[0])
# new = df1.merge(df2, on=['QuestionMention'],how='left')
new = df1[~df1.QuestionMention.isin(df2.QuestionMention)]
# print(new.shape[0])
new.drop_duplicates()
# new = new[new.Label_y.isnull()]
# total = 0
with open('log.txt', 'w') as f:
    for i in range(new.shape[0]):
        q, m = new.iloc[i]['QuestionMention'].split('--')
        _, e = new.iloc[i]['Mention_label'].split(';')
        count = df1a[df1a['QuestionMention'] == new.iloc[i]['QuestionMention']].shape[0]
        # print(count, new.iloc[i]['QuestionMention'])
        e = e.replace(' ', '_')
        f.write(f'{q}\n')
        f.write(f'      {m} : https://dbpedia.org/page/{e}      {e}\n')
        link =f'https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=PREFIX+owl%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2002%2F07%2Fowl%23%3E%0D%0APREFIX+xsd%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2001%2FXMLSchema%23%3E%0D%0APREFIX+rdfs%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2000%2F01%2Frdf-schema%23%3E%0D%0APREFIX+rdf%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0D%0APREFIX+foaf%3A+%3Chttp%3A%2F%2Fxmlns.com%2Ffoaf%2F0.1%2F%3E%0D%0APREFIX+dc%3A+%3Chttp%3A%2F%2Fpurl.org%2Fdc%2Felements%2F1.1%2F%3E%0D%0APREFIX+%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F%3E%0D%0APREFIX+dbpedia2%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fproperty%2F%3E%0D%0APREFIX+dbpedia%3A+%3Chttp%3A%2F%2Fdbpedia.org%2F%3E%0D%0APREFIX+skos%3A+%3Chttp%3A%2F%2Fwww.w3.org%2F2004%2F02%2Fskos%2Fcore%23%3E%0D%0ASELECT+*+WHERE+%7B%0D%0A%7B+%3Chttp%3A%2F%2Fdbpedia.org%2Fresource%2F{e}%3E+a+%3Ftype%7D%0D%0A%7D%0D%0A&output=json'
        f.write(f'{link}\n')
        f.write('\n\n')

# print(total)
# print(df1a[df1a['QuestionMention'] == 'Whose families are Buccinoidea and Buccinidae?--Buccinidae'])
