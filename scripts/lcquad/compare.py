import pandas as pd
fname = '../output/exp_lnn_lcquad_ensemble/prediction_Person_lcquad_gt_5000.csv'
pname = '../output/exp_lnn_lcquad_ensemble/prediction_fullPerson_lcquad_gt_5000.csv'
df1a = pd.read_csv(fname)
df2a = pd.read_csv(pname)
questions = df1a.Question.values
count = 0
for question in questions:
    result1 = eval(df1a[df1a['Question'] == question]['Entities'].values[0])
    result2 = eval(df2a[df2a['Question'] == question]['Entities'].values[0])
    if result1[0][0] != result2[0][0]:
        count+= 1
    # print(result1)
    # print(result2)
print(count / len(question))