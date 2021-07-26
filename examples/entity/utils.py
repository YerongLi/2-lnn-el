from shutil import copy
from torch.utils.data import Dataset, Sampler
import numpy as np
import pandas as pd
import torch
import sys
from tqdm import tqdm
import multiprocessing
sys.path.append('../../src/meta_rule/')
from el_evaluation import *
import copy
# from el_evaluation_redirect import *
from sklearn import preprocessing
import os
import json
benchmark =  pd.DataFrame([], columns = ['Entities', 'Classes'])
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    OKRED='\033[0;31m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tokgreen(s):
    return bcolors.OKGREEN + s + bcolors.ENDC

def tokblue(s):
    return bcolors.OKBLUE + s + bcolors.ENDC

def tokred(s):
    return bcolors.OKRED + s + bcolors.ENDC

def tokwaring(s):
    return bcolors.WARNING + s + bcolors.ENDC


class MyBatchSampler(Sampler):
    '''
    Balanced batch sampling. Assumes input consists of binary-class
    labels (0/1) and that the positive class (label=1) is the rarer
    class. Ensures that every batch consists of an equal number from
    the positive and negative class.
    '''

    def __init__(self, labels):
        self.pos_idx = list(filter(lambda i: labels[i] == 1, range(len(labels))))
        self.neg_idx = list(filter(lambda i: labels[i] == 0, range(len(labels))))

        self.pos_idx = self.pos_idx * (len(self.neg_idx) // len(self.pos_idx))  # integer division
        fillin = len(self.neg_idx) - len(self.pos_idx)
        self.pos_idx = self.pos_idx if fillin == 0 else self.pos_idx + self.pos_idx[
                                                                       0:fillin]  # pos_idx is now as long as neg_idx
        self.idx = [val for pair in zip(self.pos_idx, self.neg_idx) for val in pair]  # interleaving pos_idx and neg_idx
        self.shuffle()

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

    # call this at the end of an epoch to shuffle
    def shuffle(self):
        np.random.shuffle(self.neg_idx)
        np.random.shuffle(self.pos_idx)
        self.idx = [val for pair in zip(self.pos_idx, self.neg_idx) for val in pair]  # interleaving pos_idx and neg_idx


class QuestionSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, sampler, qm_tensors, drop_last):
        self.sampler = sampler
        self.questionMentions = qm_tensors
        self.drop_last = drop_last # not used

    def __iter__(self):
        batch = []
        prev_qm = 0
        for idx in self.sampler:
            if idx == 0 or self.questionMentions[idx] == prev_qm:
                batch.append(idx)
            else:
                prev_qm = self.questionMentions[idx]
                yield batch
                batch = [idx]
        yield batch

    def __len__(self):
        return len(torch.unique(self.questionMentions))


def convert_values_to_tensors(df_):
    db_mask_ = torch.from_numpy(df_.db.values).float().reshape(-1, 1)
    blink_mask_ = torch.from_numpy(df_.blink.values).float().reshape(-1, 1)
    qm_proc = preprocessing.LabelEncoder()
    qm_indices = qm_proc.fit_transform(df_.QuestionMention.values)
    qm_tensors_ = torch.as_tensor(qm_indices).float().reshape(-1, 1)
    return qm_tensors_, db_mask_, blink_mask_


def read_data_file(df_, device, split_name="train"):
    """
    read data file
    """

    # sort by values
    df_tmp = df_.sort_values(by=['QuestionMention', 'Label'])

    # train
    features_train = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_tmp.Features.values])
    x_tmp = torch.from_numpy(features_train).float()
    y_tmp = torch.from_numpy(df_tmp.Label.values).float().reshape(-1, 1)
    m_labels_tmp = df_tmp.Mention_label.values
    ques_tmp = df_tmp.Question.values

    # move to gpu
    x_tmp, y_tmp = x_tmp.to(device), y_tmp.to(device)

    # aggregate the data into train, val, and test
    train_data = (x_tmp, y_tmp, df_tmp)
    print(split_name, ":", x_tmp.shape, y_tmp.shape, m_labels_tmp.shape, ques_tmp.shape)

    # check class distribution
    print("y sum", sum(y_tmp), sum(y_tmp) / len(y_tmp))

    return train_data


def read_and_filter_lcquad(df_train, df_test, device):
    """
    filter out the questions with single positive or many negatives in trianing set (lcquad dataset)
    """
    # filtered_question_mentions = []
    # for qm in df_train.QuestionMention.unique():
    #     df_ = df_train[df_train.QuestionMention == qm]
    #     if df_.Label.sum() == 0:
    #         filtered_question_mentions.append(qm)
    #     if df_.Label.sum() == 1 and df_.shape[0] == 1:
    #         filtered_question_mentions.append(qm)
    # #             print(df_.Label.values)
    # df_train_split_filtered = df_train[~df_train.QuestionMention.isin(filtered_question_mentions)]
    # df_train_split_filtered = df_train_split_filtered.sort_values(by=['QuestionMention', 'Label'])
    # df_train = df_train_split_filtered

    # sort by values
    df_train = df_train.sort_values(by=['QuestionMention', 'Label'])
    df_test = df_test.sort_values(by=['QuestionMention', 'Label'])

    # train
    features_train = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_train.Features.values])
    x_train = torch.from_numpy(features_train).float()
    y_train = torch.from_numpy(df_train.Label.values).float().reshape(-1, 1)
    m_labels_train = df_train.Mention_label.values
    ques_train = df_train.Question.values

    # test
    features_test = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_test.Features.values])
    x_test = torch.from_numpy(features_test).float()
    y_test = torch.from_numpy(df_test.Label.values).float().reshape(-1, 1)
    m_labels_test = df_test.Mention_label.values
    ques_test = df_test.Question.values

    # move to gpu
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    # aggregate the data into train, val, and test
    train_data = (x_train, y_train, df_train)
    print("train:", x_train.shape, y_train.shape, m_labels_train.shape, ques_train.shape)
    test_data = (x_test, y_test, df_test)
    print("test:", x_test.shape, y_test.shape, m_labels_test.shape, ques_test.shape)

    # check class distribution
    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    return train_data, test_data


def compute_qald_metrics(pred_, df_, gold_file_name='lcquad_gt_5000.csv', topk=5, is_long = False):
    """pred_ are 0/1 s after applying a threshold"""
    n, l, r = df_.shape[0], 0, 0
    data = []
    try:
        while r < n:
            while r < n - 1 and df_.iloc[l]['QuestionMention'] == df_.iloc[r + 1]['QuestionMention']:
                r+= 1
            batch = df_.iloc[l : r + 1]
            gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
            # assert(len(gold_pairs) == 1)
            gt = gold_pairs[0].split(';')[1].replace(' ', '_')
            j = torch.argmax(pred_[l:r + 1]).numpy()
            # print(j)
            if j != r - l:
                data.append([df_.iloc[l]['QuestionMention'],
                f'https://dbpedia.org/page/{gt}', 
                f"https://dbpedia.org/page{batch.iloc[j]['Mention_label'].split(';')[1].replace(' ', '_')}"])
            l = r + 1
            r = l
        error_df =  pd.DataFrame(data, columns = ['QuestionMention','gold','pred'])
        error_df.to_csv('error.csv', index=False)
    except:
        pass

    ques_ = df_.Question.values
    m_labels_ = df_.Mention_label.values
    labels_ = df_.Label.values

    rows = []
    question_rows_map = {}
    question_mention_set = set()
    for i, pred in enumerate(pred_):
        pred = pred.data.tolist()[0]
        question = ques_[i]
        if question not in question_rows_map:
            question_rows_map[ques_[i]] = []
        if pred:
            men_entity_label = '_'.join(m_labels_[i].split(';')[-1].split())
            men_entity_mention = '_'.join(m_labels_[i].split(';')[0].split())
            if '-'.join([question, men_entity_mention]) in question_mention_set:
                question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
            else:
                question_mention_set.add('-'.join([question, men_entity_mention]))
                question_rows_map[ques_[i]].append(set())
                question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
    for key, preds_list_mentions in question_rows_map.items():
        # print(preds_list_mentions)
        if len(preds_list_mentions) > 1:
            rows.append([key, []])
            for preds_set in preds_list_mentions:
                sorted_values = sorted(list(preds_set), key=lambda x: (-x[1], x[2]))[:topk]
                rows[-1][1].append(sorted_values)
        elif len(preds_list_mentions) == 1:
            sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: (-x[1], x[2]))[:topk]
            rows.append([key, [sorted_values]])
        else:
            rows.append([key, []])
    # print('rows', rows)
    # print(question_rows_map)
    df_output = pd.DataFrame(rows, columns=['Question', 'Entities'])
    df_output['Classes'] = str([])

    # gold
    benchmark = pd.read_csv(gold_file_name)
    benchmark = benchmark.set_index('Question')
    benchmark = benchmark.replace(np.nan, '', regex=True)
    benchmark['Entities'] = benchmark['Entities'].astype(object)
    is_qald_gt = True

    # pred
    predictions = df_output
    # print(df_output.shape)
    predictions = predictions.set_index('Question')
    predictions['Entities'] = predictions['Entities']
    predictions['Classes'] = predictions['Classes']

    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')
    # metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')

    scores = metrics['macro']['named']
    # print(scores['micro']['named']['f1'])
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output

def compute_qald_metricsList(pred_, df_, topk=5, printerror=False):
    """pred_ are 0/1 s after applying a threshold"""

    global benchmark
    separator = '===' if '===' in df_.iloc[0].Mention_label else ';'
    # benchmark = pd.read_csv(gold_file_name)
    # benchmark = benchmark.set_index('Question')
    # benchmark = benchmark.replace(np.nan, '', regex=True)
    # benchmark['Entities'] = benchmark['Entities'].astype(object)
    df_['Question'] = df_['Question'].astype('string')
    # df_.to_csv('df.csv')

    if benchmark.shape[0] == 0 or not set(df_.iloc[:5].Question.values).issubset(set(benchmark.index.values)):
        print(tokwaring("Appending to benchmark"), end='\r')
        n, l, r = df_.shape[0], 0, 0
        benchmarkdata = dict()
        # index = []
        while r < n:
            while r < n - 1 and df_.iloc[l]['Question'] == df_.iloc[r + 1]['Question']:
                r+= 1
            batch = df_.iloc[l : r + 1]
            gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
            # print(separator)
            # print(gold_pairs)
            gt = [gp.split(separator)[1].replace(' ', '_') for gp in gold_pairs]
            # print('len gt', len(gt))
            if batch.iloc[0].Question not in benchmarkdata:
                benchmarkdata[batch.iloc[0].Question] = []
                # index.append(batch.iloc[0].Question)
            benchmarkdata[batch.iloc[0].Question].extend(gt)
            # j = torch.argmax(pred_[l:r + 1]).numpy()
            # # print(j)
            # if j != r - l:
            #     data.append([df_.iloc[l]['QuestionMention'],
            #     f'https://dbpedia.org/page/{gt}', 
            #     f"https://dbpedia.org/page{batch.iloc[j]['Mention_label'].split(';')[1].replace(' ', '_')}"])
            l = r + 1
            r = l
        benchmarkdata = [[i, str(benchmarkdata[i]), str([])] for i in benchmarkdata.keys()]
        newbenchmark =  pd.DataFrame(benchmarkdata, columns = ['Question','Entities', 'Classes'])
        newbenchmark['Question'] = newbenchmark['Question'].astype('string')

        newbenchmark = newbenchmark.set_index('Question')
        newbenchmark = newbenchmark.replace(np.nan, '', regex=True)
        newbenchmark['Entities'] = newbenchmark['Entities'].astype(object)
        benchmark = benchmark.append(newbenchmark)
        print(tokwaring('Extended benchmark     '))
        del newbenchmark
#         # error_df.to_csv('error.csv', index=False)
    # print(benchmark)        
    ques_ = df_.Question.values
    m_labels_ = df_.Mention_label.values
    labels_ = df_.Label.values
    qm_ = df_.QuestionMention.values

    # index = []
    rows = []
    question_rows_map = dict()
    benchmarkdata = dict()
    for i, pred in enumerate(pred_):
        pred = pred.data.tolist()[0]
        if ques_[i] not in question_rows_map:
            # index.append(ques_[i])
            question_rows_map[ques_[i]] = []
            benchmarkdata[ques_[i]] = []
            

        men_entity_label = '_'.join(m_labels_[i].split(separator)[-1].split())

        # assert(labels_[i] == 1 or labels_[i] == 0)
        # if labels_[i] == 1:
            # benchmarkdata[ques_[i]].append(men_entity_label)
        if pred: 
            men_entity_mention = '_'.join(m_labels_[i].split(separator)[0].split())
            # print([question, men_entity_mention])
            # if '-'.join([question, men_entity_mention]) in question_mention_set:
            # if qm_[i] in question_mention_set:

            if i > 0 and qm_[i] == qm_[i-1]:
                if 0==len(question_rows_map[ques_[i]]):
                    question_rows_map[ques_[i]].append(set())
                # print(question_rows_map[ques_[i]])
                question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_mention))
            else:
                # question_mention_set.add(qm_[i])
                question_rows_map[ques_[i]].append(set())
                question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label, men_entity_mention))
    predrows = []
    for key, preds_list_mentions in question_rows_map.items():
        # print(preds_list_mentions)
        # print(key)
        if len(preds_list_mentions) > 1:
            predrows.append([key, []])
            rows.append([key, []])
            for preds_set in preds_list_mentions:
                sorted_values = sorted(list(preds_set), key=lambda x: (-x[1], x[2]))[:topk]
                predrows[-1][1].append([sorted_values[0][2], sorted_values[0][2]])
                rows[-1][1].append(sorted_values)
        elif len(preds_list_mentions) == 1:
            # sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: (-x[1], x[2]))[:topk]
            # print(list(preds_list_mentions))
            predrows.append([key, [list(preds_list_mentions[0])[0][2],list(preds_list_mentions[0])[0][2]]])
            rows.append([key, [list(preds_list_mentions[0])]])
            # print(tokgreen(key), 'length 1')
        else:
            predrows.append([key, []])
            rows.append([key, []])
    # # print(rows, 'rows')
    # if benchmark.shape[0] == 0 or not set(df_.iloc[:5].Question.values).issubset(set(benchmark.index.values)):
    #     print(tokwaring("Appending to benchmark"))
    #     benchmarkdata = [[i, str(benchmarkdata[i]), str([])] for i in index]
    #     newbenchmark =  pd.DataFrame(benchmarkdata, columns = ['Question','Entities', 'Classes'])
    #     newbenchmark['Question'] = newbenchmark['Question'].astype('string')

    #     newbenchmark = newbenchmark.set_index('Question')
    #     newbenchmark = newbenchmark.replace(np.nan, '', regex=True)
    #     newbenchmark['Entities'] = newbenchmark['Entities'].astype(object)
    #     benchmark = benchmark.append(newbenchmark)
    #     del newbenchmark
    # print('rows', rows)
    # print(question_rows_map)

    df_output = pd.DataFrame(predrows, columns=['Question', 'Entities'])
    df_output['Classes'] = str([])
    df_output.to_csv('pred.csv', index=False)

    # gold

    is_qald_gt = True

    # pred
    predictions = pd.DataFrame(rows, columns=['Question', 'Entities'])
    predictions['Classes'] = str([])
    # print(df_output.shape)
    predictions['Question'] = predictions['Question'].astype('string')
    predictions = predictions.set_index('Question')

    predictions['Entities'] = predictions['Entities']
    predictions['Classes'] = predictions['Classes']
    count = 0
    if printerror:
        # for doc in ['1320testb']:
        for doc in predictions.index.values.tolist():
            pr = predictions.loc[doc].Entities
            # print(pr)
            pr = [p[0][0].split('/')[-1] for p in pr]
            gd = eval(benchmark.loc[doc].Entities)
            # if doc == '1164testb':
            if len(pr) != len(gd):
                print(predictions.loc[doc].Entities)
                print(tokblue(doc), len(pr), len(gd))
                print('pr', pr)
                print('gd', gd)
            count+= 1
    
        # print(benchmark.loc)
    # print(f'mismatch {count}/{len(predictions.index.values.tolist())}')
    # print(benchmark)
    # print(predictions)
    # print(printerror)
    metrics = compute_metricsList(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full', printerror=printerror)
    # metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')

    scores = metrics['macro']['named']
    # print(scores['micro']['named']['f1'])
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output

def get_prediction(docinput):
    (doc, pred, df) = docinput
    assert(len(pred) == df.shape[0])
    if df.shape[0]==0: return None
    # print(set(df.Doc.values.tolist()))
    # assert(len(set(df.Doc.values.tolist()))==1)
    # assert(doc==df.Doc.values.tolist()[0])
    n, l, r = df.shape[0], 0, 0
    result = []
    golds = []
    mns = []
    while r < n:
        while r < n - 1 and df.iloc[l]['QuestionMention'] == df.iloc[r + 1]['QuestionMention']:
            r+= 1
        batch = df.iloc[l : r + 1]
        gold = batch[batch.Label.eq(1)].iloc[0].Mention_label.split('===')[1]
        mn = batch[batch.Label.eq(1)].iloc[0].Mention_label.split('===')[0]
        mns.append(mn)
        golds.append(gold)
        # assert(len(gold_pairs) == 1)
        # gt = gold_pairs[0].split(';')[1].replace(' ', '_')
        j = np.argmax(np.array(pred[l:r + 1]))
        # if doc == '1173testb':
        #     print(batch.Mention_label)
        #     print('j =', j, pred[l:r + 1])
        result.append(batch.iloc[j].Mention_label.split('===')[1])
        l = r + 1
        r = l
    # print(result)
    # if doc == '1173testb':
    #     print('res',result)
    #     print('gol',golds)
    return (doc, result, golds, mns)

def evaluation(testset, system_pred):
    gold = []
    pred = []
    # entity2type = pkl.load(open(os.getenv("HOME") + '/data/entity2type.pkl', 'rb'))
    tjson = json.load(open(os.getenv("HOME") + '/DCA/entityType.json', 'r'))
    # print(tjson)
    for doc_name, content in testset.items():
        if doc_name not in system_pred: continue
        # print(doc_name)
        # print(len(content), len(system_pred[doc_name]))
        # print(content)
        # print(system_pred[doc_name])
        # print(len(content) == len(system_pred[doc_name]))
        # if doc_name == '1287testb':
        #     print('predictions', doc_name)
        #     print(content)
        #     print([c for c in system_pred[doc_name]])
        gold += content
        pred += [c for c in system_pred[doc_name]]
    true_pos = 0
    matrix = [[0,0,0,0] for _ in range(4)]
    errortotal = 0

    for g, p in zip(gold, pred):
        gtype = tjson[g.replace(' ', '_')] if g.replace(' ', '_') in tjson else 3
        ptype = tjson[p.replace(' ', '_')] if p.replace(' ', '_') in tjson else 3
        if gtype != ptype:
            matrix[gtype][ptype]+= 1
        if g == p and p != 'NIL':
            true_pos += 1
        else:
            errortotal+= 1
    s = 0
    for i in range(4):
        for j in range(4):
            s += matrix[i][j]
            matrix[i][j] = matrix[i][j] / len(gold)
            print(round(matrix[i][j],4), end =' ')
        print()
    print(s/ errortotal)
    # print(matrix)

    precision = true_pos / len([p for p in pred if p != 'NIL'])
    recall = true_pos / len(gold)
    f1 = 0 if true_pos ==0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def compute_long_metrics(pred_, df_, gold_file_name='lcquad_gt_5000.csv', topk=5, is_long = False):
    prec, recall, f1, df_output = None, None, None, None
    # gold
    # benchmark = pd.read_csv(gold_file_name)
    # print(df_.shape[0], len(pred_))
    pred_ = [s[0] for s in pred_.numpy()]
    # print(pred_)
    # print(len(pred_), df_.shape[0])
    assert(len(pred_) == df_.shape[0])

    documents = set(df_.Doc.values.tolist())
    # print(documents)

    docinput = []
    for doc in documents:
        batchi = df_.index[df_['Doc']==doc].tolist()
        if not batchi: continue
        # print(doc)
        le = min(batchi)
        ri = max(batchi)
        # b = copy.copy(df_.loc[le:ri])
        b = df_[df_['Doc']==doc]

        # print(set(b.Doc.values.tolist()))

        # print(le, ri)
        # print(df_.loc[le].Doc, le)
        # print(df_.loc[ri].Doc, ri)
        # print(b[b.Doc == '948testa'])

        assert(1==len(set(b.Doc.values.tolist())))
        docinput.append((doc, pred_[le:ri + 1], b))
    # print('docinput', docinput)
    with multiprocessing.Pool(40) as pool:
        predictions = list(pool.map(get_prediction, docinput))
    predictions_temp = [content for content in predictions if content is not None]
    # print(list(predictions_temp))
    key = str(predictions_temp[0][0])
    mns = {str(content[0]) : content[3] for content in predictions_temp}
    predictions = {str(content[0]) : content[1] for content in predictions_temp}
    testset = {str(content[0]) : content[2] for content in predictions_temp}
    
    # print(predictions.keys())
    # testset ={str(benchmark.iloc[i].Question): eval(benchmark.iloc[i].gold) for i in range(benchmark.shape[0])}
    print(key)
    print('mentions', mns[key])
    print(predictions[key], len(predictions[key]))
    print(testset[key],len(predictions[key]))
    prec, recall, f1 = evaluation(testset, predictions)
    df_output = pd.DataFrame([], columns=['Question', 'Entities'])

    return prec, recall, f1, df_output

def compute_qald_metrics_hybrid(pred_dict, df_, gold_file_name='lcquad_gt_5000.csv', topk=5):
    """pred_ are 0/1 s after applying a threshold"""

   
    ques_ = df_.Question.values
    qm_ = df_.QuestionMention.values
    m_labels_ = df_.Mention_label.values

    rows = []
    question_rows_map = {}
    question_mention_set = set()
    # for name in ['Person']:
    print(pred_dict.keys())
    for name in pred_dict:
        if name == 'full':
            continue
        pred_ = pred_dict[name]
        type_df = None if name == 'full' else pd.read_csv(f"data/lcquad/blink_bert_box/{name}_test.csv")
        for i, pred in tqdm(enumerate(pred_)):
            pred = pred.data.tolist()[0]
            question = ques_[i]
            quesionMention= qm_[i]
            if type_df is not None:
                find_df = type_df[type_df.QuestionMention ==quesionMention]
                if find_df.shape[0] == 0:
                    continue
            if question not in question_rows_map:
                question_rows_map[ques_[i]] = dict()
            if pred:
                men_entity_label = '_'.join(m_labels_[i].split(';')[-1].split())
                men_entity_mention = '_'.join(m_labels_[i].split(';')[0].split())
                if '-'.join([question, men_entity_mention]) in question_mention_set:
                    question_rows_map[ques_[i]][men_entity_mention].add(
                        ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
                else:
                    if ques_[i] not in question_rows_map:
                        question_rows_map[ques_[i]] = dict()
                    question_mention_set.add('-'.join([question, men_entity_mention]))
                    question_rows_map[ques_[i]][men_entity_mention] = set()
                    question_rows_map[ques_[i]][men_entity_mention].add(
                        ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
        del type_df
        # pred_ = pred_dict[name]
    typed_question_mention_set= copy.deepcopy(question_mention_set)
    question_mention_set = set()
    for i, pred in tqdm(enumerate(pred_dict['full'])):
        pred = pred.data.tolist()[0]
        question = ques_[i]
        quesionMention= qm_[i]
        if question not in question_rows_map:
            question_rows_map[ques_[i]] = dict()
        if pred:
            men_entity_label = '_'.join(m_labels_[i].split(';')[-1].split())
            men_entity_mention = '_'.join(m_labels_[i].split(';')[0].split())
            if '-'.join([question, men_entity_mention]) in typed_question_mention_set:
                continue
            if '-'.join([question, men_entity_mention]) in question_mention_set:
                # continue
                question_rows_map[ques_[i]][men_entity_mention].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
            else:
                if ques_[i] not in question_rows_map:
                    question_rows_map[ques_[i]] = dict()
                question_mention_set.add('-'.join([question, men_entity_mention]))
                question_rows_map[ques_[i]][men_entity_mention] = set()
                question_rows_map[ques_[i]][men_entity_mention].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))

# 
    for key, preds_dict_mentions in question_rows_map.items():
        preds_list_mentions = list(preds_dict_mentions.values())
        # print(preds_list_mentions)
        if len(preds_list_mentions) > 1:
            rows.append([key, []])
            for preds_set in preds_list_mentions:
                sorted_values = sorted(list(preds_set), key=lambda x: (-x[1], x[2]))[:topk]
                rows[-1][1].append(sorted_values)
        elif len(preds_list_mentions) == 1:
            sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: (-x[1], x[2]))[:topk]
            rows.append([key, [sorted_values]])
        else:
            rows.append([key, []])
    # print('rows', rows)
    # print(question_rows_map)
    df_output = pd.DataFrame(rows, columns=['Question', 'Entities'])
    df_output['Classes'] = str([])

    # gold
    benchmark = pd.read_csv(gold_file_name)
    benchmark = benchmark.set_index('Question')
    benchmark = benchmark.replace(np.nan, '', regex=True)
    benchmark['Entities'] = benchmark['Entities'].astype(object)
    is_qald_gt = True

    # pred
    predictions = df_output
    # print(df_output.shape)
    predictions = predictions.set_index('Question')
    predictions['Entities'] = predictions['Entities']
    predictions['Classes'] = predictions['Classes']

    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')
    # metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')

    scores = metrics['macro']['named']
    # print(scores['micro']['named']['f1'])
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output


def compare(pred_list, df_, gold_file_name='lcquad_gt_5000.csv', topk=5):
    """pred_ are 0/1 s after applying a threshold"""

    # print(df_.keys())
    qm_ = df_.QuestionMention.values
    m_labels_ = df_.Mention_label.values

    rows = []
    question_rows_map = {}
    question_mention_set = set()
    # print(pred_list[0].shape)
    # print(pred_list[1].shape)
    pred_ = pred_list[0]
    pass
    for i in range(pred_list[0].shape[0]):
        pred = [pred_list[0][i].data.tolist()[0],  pred_list[1][i].data.tolist()[0]]
        qm = qm_[i]
        print(qm)
        # if question not in question_rows_map:
        #     question_rows_map[ques_[i]] = []
        if True:
            men_entity_label = '_'.join(m_labels_[i].split(';')[-1].split())
            men_entity_mention = '_'.join(m_labels_[i].split(';')[0].split())
            question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
            # if '-'.join([question, men_entity_mention]) in question_mention_set:
            #     question_rows_map[ques_[i]][-1].add(
            #         ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))
            # else:
            #     question_mention_set.add('-'.join([question, men_entity_mention]))
            #     question_rows_map[ques_[i]].append(set())
            #     question_rows_map[ques_[i]][-1].add(
            #         ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred, men_entity_label))


    # for key, preds_list_mentions in question_rows_map.items():
    #     if len(preds_list_mentions) > 1:
    #         rows.append([key, []])
    #         for preds_set in preds_list_mentions:
    #             sorted_values = sorted(list(preds_set), key=lambda x: (-x[1], x[2]))[:topk]
    #             rows[-1][1].append(sorted_values)
    #     elif len(preds_list_mentions) == 1:
    #         sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: (-x[1], x[2]))[:topk]
    #         rows.append([key, [sorted_values]])
    #     else:
    #         rows.append([key, []])
    # # print('rows', rows)
    # df_output = pd.DataFrame(rows, columns=['Question', 'Entities'])
    # df_output['Classes'] = str([])

    # # gold
    # benchmark = pd.read_csv(gold_file_name)
    # benchmark = benchmark.set_index('Question')
    # benchmark = benchmark.replace(np.nan, '', regex=True)
    # benchmark['Entities'] = benchmark['Entities'].astype(object)
    # is_qald_gt = True

    # # pred
    # predictions = df_output
    # # print(df_output.shape)
    # predictions = predictions.set_index('Question')
    # predictions['Entities'] = predictions['Entities']
    # predictions['Classes'] = predictions['Classes']

    # metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')
    # # metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval='full')

    # scores = metrics['macro']['named']
    # # print(scores['micro']['named']['f1'])
    # prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    # return prec, recall, f1, df_output