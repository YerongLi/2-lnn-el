import pandas as pd
import numpy as np
from argparse import ArgumentParser 
import ast
import pickle
import json
import os
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

def new_overlap(gold,pred):
    for p1 in pred:
        if any(p1.strip(' ') == u for u in gold):
            return 0
    return 1
def compute_overlap(gold,pred, debug=False):
    correct = []
    for i in gold:
        if any(i.strip(' ') == u for u in pred):
            correct.append(i)
        elif debug:
            print("missing {} {}".format(i, pred))
    if len(gold) == 0:
        correct_percent = 100
    else:
        correct = list(dict.fromkeys(correct))
        correct_percent = len(correct)*100/len(gold)
    return correct, correct_percent

def compute_overlapList(gold,pred, debug=False):
    # debug = False
    # print(gold)
    # print(pred)
    # print(len(gold), len(pred))

    if debug and gold:
        print(len(gold), len(pred))
    if len(gold)!=len(pred):
        print('================',len(gold), len(pred))
        print(gold)
        print(pred)
    assert(len(gold)==len(pred))
    correct = []
    for g, p in zip(gold, pred):
        if g.strip(' ') == p:
            correct.append(g)
        # elif debug:
            # print("missing {} {}".format(g, pred))
    if debug and gold:
        goldlist = [tokgreen(g) for i, g in enumerate(gold) if not g.strip(' ') == pred[i]]
        predlist = [tokred(pred[i]) for i, g in enumerate(gold) if not g.strip(' ') == pred[i]]
        for i in range(len(goldlist)):
            print(goldlist[i], predlist[i])
        # print('\t'.join(goldlist))
        # print('\t'.join(predlist))
    if len(gold) == 0:
        correct_percent = 100
    else:
        correct = correct
        correct_percent = len(correct)*100/len(gold)
        mistakelist = [[g, pred[i]]  for i, g in enumerate(gold) if not g.strip(' ') == pred[i]]
    return correct, correct_percent, mistakelist

def get_prec_rec(tp=0, fp=0, fn=0):
    if fp > 0:
        precision = tp*1.0/(tp+fp)
    else:
        precision = 1.0
    if fn > 0:
        recall = tp*1.0/(tp+fn)
    else:
        recall = 1.0 
    return precision, recall
def get_f1(p=0.0, r=0.0):
    if p > 0 or r > 0: 
        return 2*p*r/(p+r)
    else:
        return 0.0

def compute_metrics(benchmark=None, predictions=None, limit=250, is_qald_gt=False, eval='full'):
    TP_R, TP_O, TP_C, FP_R, FP_O, FP_C, FN_R, FN_O, FN_C = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0
    named_p, named_r, nominal_p, nominal_r, combined_p, combined_r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sent_count = 0
    metrics = {}

    ont_prefix = 'http://dbpedia.org/ontology/'
    res_prefix = 'http://dbpedia.org/resource/'

    named_sent_count = 0
    nominal_sent_count = 0
    combined_sent_count = 0
    skip_count = 0
    data = []
    if benchmark is None or predictions is None:
        print("Need both benchmark and predicitons to compute metrics")
        return metrics

    for index, gold_row in benchmark.iterrows():
        if eval == 'dev1' and (sent_count < 250 or sent_count >= 350):
            sent_count += 1
            continue
        if eval == 'dev2' and (sent_count < 350):
            sent_count += 1
            continue
        
        #if sent_count <= 100:
        #    sent_count += 1
        #    continue
        
        # get nominal and named entities predictions
        if index not in predictions.index:
            skip_count += 1
            continue
        print(predictions.loc[index]['Entities'])
        pred_entities = ast.literal_eval(str(predictions.loc[index]['Entities']))

        pred_nominal = [p[0][0].replace(ont_prefix, '') for p in pred_entities if 'ontology' in p[0][0]]
        
        pred_named = [p[0][0].replace(res_prefix, '') for p in pred_entities if 'resource' in p[0][0]]
        pred_combined = []
        pred_combined.extend(pred_named)
        pred_combined.extend(pred_nominal)

        # (Almaden's groundtruth)
        # get gold nominal and named entities
        try:
            if 0 == len(ast.literal_eval(gold_row['Entities'])):
                skip_count += 1
                continue   
            if not is_qald_gt:
                gold_entities = gold_row['Entities'].replace('[','').replace(']','').replace('\'','')
                gold_entities = list(gold_entities.split(','))
                gold_classes = gold_row['Classes'].replace('[','').replace(']','').replace('\'','')
                gold_classes = list(gold_classes.split(','))
                gold_entities.extend(gold_classes)
                gold_nominal = [p.replace(ont_prefix, '').strip() for p in gold_entities if 'ontology' in p]
                gold_named = [p.replace(res_prefix, '').strip() for p in gold_entities if 'resource' in p]
            else:
            # (QALD SPARQL's groundtruth)
                gold_entities = ast.literal_eval(gold_row['Entities'])
                gold_entities = [e.replace('\t', "'") for e in gold_entities]
                gold_classes = ast.literal_eval(gold_row['Classes'])
                gold_named = [p.replace('res:', '').strip() for p in gold_entities]
                gold_nominal = [p.replace('onto:','').replace('.','').replace('dbo:','').strip() for p in gold_classes]

            gold_combined = []
            gold_combined.extend(gold_named)
            gold_combined.extend(gold_nominal)
            gold_combined = sorted(gold_combined)
        except:
            # import traceback
            # print(gold_row['Entities'])
            # print(0 == len(gold_row['Entities']))
            # traceback.print_exc()
            skip_count += 1
            continue   

        #if index == 'What has Carl Sagan written his books about?':
        #print(gold_named)
        #print('>>', pred_named)
        
        entities_correct,_ = compute_overlap(gold_named,pred_named, debug=False)
        classes_correct,_ = compute_overlap(gold_nominal,pred_nominal, debug=False)
        combined_correct,correct_percent = compute_overlap(gold_combined,pred_combined)
        # print(gold_row.keys())
        if correct_percent < 99.9:
            data.append([index, gold_combined, pred_combined])

        #print("{} {} {}".format(entities_correct, classes_correct, combined_correct))

        # per sentence 'resource' metrics
        if True or len(gold_named):
            named_sent_count += 1
            tp_r = len(entities_correct)
            fp_r = len(pred_named) - tp_r
            fn_r = len(gold_named) - tp_r
            TP_R += tp_r
            FP_R += fp_r
            FN_R += fn_r
            p,r =  get_prec_rec(tp=tp_r, fp=fp_r, fn=fn_r)
            named_p += p
            named_r += r

        # per sentence 'ontology' metrics
        if True or len(gold_nominal):
            nominal_sent_count += 1
            tp_o = len(classes_correct)
            fp_o = len(pred_nominal) - tp_o
            fn_o = len(gold_nominal) - tp_o
            TP_O += tp_o
            FP_O += fp_o
            FN_O += fn_o
            p,r =  get_prec_rec(tp=tp_o, fp=fp_o, fn=fn_o)
            nominal_p += p
            nominal_r += r
        
        # per sentence combined metrics
        if True or len(gold_combined):
            combined_sent_count += 1
            tp_c = len(combined_correct)
            fp_c = len(pred_combined) - tp_c
            fn_c = len(gold_combined) - tp_c
            TP_C += tp_c
            FP_C += fp_c
            FN_C += fn_c
            p,r =  get_prec_rec(tp=tp_c, fp=fp_c, fn=fn_c)
            combined_p += p
            combined_r += r

       # print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))

        sent_count +=1

        #if sent_count == 1:
        #    break
    error_df =  pd.DataFrame(data, columns = ['Question','gold','pred'])
    error_df.to_csv('error_old.csv', index=False)


    #macros
    #print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))
    #print("{} {} {} {}".format(named_sent_count, nominal_sent_count, combined_sent_count, skip_count))
    if named_sent_count == 0:
        named_sent_count += 1
    macro_named_p = named_p/named_sent_count
    macro_named_r = named_r/named_sent_count
    macro_named_f1 = get_f1(p=macro_named_p, r=macro_named_r)


    if nominal_sent_count == 0:
        nominal_sent_count += 1
    macro_nominal_p = nominal_p/nominal_sent_count
    macro_nominal_r = nominal_r/nominal_sent_count
    macro_nominal_f1 = get_f1(p=macro_nominal_p, r=macro_nominal_r)

    if combined_sent_count == 0:
        combined_sent_count += 1
    macro_combined_p = combined_p/combined_sent_count
    macro_combined_r = combined_r/combined_sent_count
    macro_combined_f1 = get_f1(p=macro_combined_p, r=macro_combined_r)

    
    # micros
    micro_named_p, micro_named_r = get_prec_rec(tp=TP_R, fp=FP_R, fn=FN_R)
    micro_named_f1 = get_f1(p=micro_named_p, r=micro_named_r)

    micro_nominal_p, micro_nominal_r = get_prec_rec(tp=TP_O, fp=FP_O, fn=FN_O)
    micro_nominal_f1 = get_f1(p=micro_nominal_p, r=micro_nominal_r)

    micro_combined_p, micro_combined_r = get_prec_rec(tp=TP_C, fp=FP_C, fn=FN_C)
    micro_combined_f1 = get_f1(p=micro_combined_p, r=micro_combined_r)

    nominal = {'precision': round(macro_nominal_p,4), 'recall': round(macro_nominal_r,4), 'f1': round(macro_nominal_f1,4)}
    named = {'precision': round(macro_named_p,4), 'recall': round(macro_named_r,4), 'f1': round(macro_named_f1,4)}
    combined = {'precision': round(macro_combined_p,4), 'recall': round(macro_combined_r,4), 'f1': round(macro_combined_f1,4)}
    macro = {'nominal':nominal, 'named': named, 'combined': combined}

    nominal = {'precision': round(micro_nominal_p,4), 'recall': round(micro_nominal_r,4), 'f1': round(micro_nominal_f1,4)}
    named = {'precision': round(micro_named_p,4), 'recall': round(micro_named_r,4), 'f1': round(micro_named_f1,4)}
    combined = {'precision': round(micro_combined_p,4), 'recall': round(micro_combined_r,4), 'f1': round(micro_combined_f1,4)}
    
    micro = {'nominal':nominal, 'named': named, 'combined': combined}

    metrics['macro'] = macro
    metrics['micro'] = micro
    #print(metrics)
    # print('Skip Que Count',skip_count)
    return metrics

def compute_metricsList(benchmark=None, predictions=None, limit=250, is_qald_gt=False, eval='full',printerror=False):
    TP_R, TP_O, TP_C, FP_R, FP_O, FP_C, FN_R, FN_O, FN_C = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0
    named_p, named_r, nominal_p, nominal_r, combined_p, combined_r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sent_count = 0
    metrics = {}

    # print(benchmark[benchmark.Question == 'AH141 highway is a part of which country?'])
    
    # print('benchmark', benchmark)
    ont_prefix = 'http://dbpedia.org/ontology/'
    res_prefix = 'http://dbpedia.org/resource/'

    named_sent_count = 0
    nominal_sent_count = 0
    combined_sent_count = 0
    skip_count = 0
    data = []
    if benchmark is None or predictions is None:
        print("Need both benchmark and predicitons to compute metrics")
        return metrics
    matrix = [[0,0,0,0] for _ in range(4)]
    lengold = 0
    errortotal = 0
    tjson = json.load(open(os.getenv("HOME") + '/DCA/entityType.json', 'r'))

    for index, gold_row in benchmark.iterrows():
        if eval == 'dev1' and (sent_count < 250 or sent_count >= 350):
            sent_count += 1
            continue
        if eval == 'dev2' and (sent_count < 350):
            sent_count += 1
            continue
        
        #if sent_count <= 100:
        #    sent_count += 1
        #    continue
        
        # get nominal and named entities predictions
        if index not in predictions.index:
            skip_count += 1
            continue
        # print(index)
        # print(predictions.loc[index]['Entities'])
    
        pred_entities = ast.literal_eval(str(predictions.loc[index]['Entities']))

        pred_nominal = [p[0][0].replace(ont_prefix, '') for p in pred_entities if 'ontology' in p[0][0]]
        
        pred_named = [p[0][0].replace(res_prefix, '') for p in pred_entities if 'resource' in p[0][0]]
        pred_combined = []
        pred_combined.extend(pred_named)
        pred_combined.extend(pred_nominal)

        # (Almaden's groundtruth)
        # get gold nominal and named entities
        try:
            if 0 == len(ast.literal_eval(gold_row['Entities'])):
                skip_count += 1
                continue   
            if not is_qald_gt:
                gold_entities = gold_row['Entities'].replace('[','').replace(']','').replace('\'','')
                gold_entities = list(gold_entities.split(','))
                gold_classes = gold_row['Classes'].replace('[','').replace(']','').replace('\'','')
                gold_classes = list(gold_classes.split(','))
                gold_entities.extend(gold_classes)
                gold_nominal = [p.replace(ont_prefix, '').strip() for p in gold_entities if 'ontology' in p]
                gold_named = [p.replace(res_prefix, '').strip() for p in gold_entities if 'resource' in p]
            else:
            # (QALD SPARQL's groundtruth)
                gold_entities = ast.literal_eval(gold_row['Entities'])
                gold_entities = [e.replace('\t', "'") for e in gold_entities]
                gold_classes = ast.literal_eval(gold_row['Classes'])
                gold_named = [p.replace('res:', '').strip() for p in gold_entities]
                gold_nominal = [p.replace('onto:','').replace('.','').replace('dbo:','').strip() for p in gold_classes]

            gold_combined = []
            gold_combined.extend(gold_named)
            gold_combined.extend(gold_nominal)
            # gold_combined = sorted(gold_combined)
        except:
            import traceback
            # print(gold_row['Entities'])
            # print(0 == len(gold_row['Entities']))
            traceback.print_exc()
            skip_count += 1
            continue
        if printerror:
            print(tokblue(index))
        # if index == '1164testb':
        #     print(tokblue(index))
        #     print(gold_named)
        #     print(pred_named)
        # entities_correct,_ = compute_overlap(gold_named,pred_named, debug=False)
        entities_correct,_,mistakelst = compute_overlapList(gold_named,pred_named, debug=False)
        # print('entities_correct' , entities_correct)
        lengold+= len(gold_named)
        errortotal+= len(mistakelst)
        for item in mistakelst:
            gtype = tjson[item[0].replace(' ', '_')] if item[0].replace(' ', '_') in tjson else 3
            ptype = tjson[item[1].replace(' ', '_')] if item[1].replace(' ', '_') in tjson else 3
            if gtype != ptype:
                matrix[gtype][ptype]+= 1
        # if g == p and p != 'NIL':
        #     true_pos += 1
        # else:
        #     errortotal+= 1
        # classes_correct,_ = compute_overlap(gold_nominal,pred_nominal, debug=False)
        # classes_correct,_,_ = compute_overlapList(gold_nominal,pred_nominal, debug=False)
        
        # print(printerror)
        # combined_correct,correct_percent = compute_overlap(gold_combined,pred_combined)
        combined_correct,correct_percent, _ = compute_overlapList(gold_combined,pred_combined,printerror)
        # print(gold_row.keys())
        if correct_percent < 99.9:
            data.append([index, gold_combined, pred_combined])

        #print("{} {} {}".format(entities_correct, classes_correct, combined_correct))

        # per sentence 'resource' metrics
        if True or len(gold_named):
            named_sent_count += 1
            tp_r = len(entities_correct)
            fp_r = len(pred_named) - tp_r
            fn_r = len(gold_named) - tp_r
            # print(tokgreen('entities_correct'), entities_correct)
            # print(pred_named)
            # print(gold_named)
            # print(len(entities_correct))
            # print(len(pred_named))
            # print(len(gold_named))

            TP_R += tp_r
            FP_R += fp_r
            FN_R += fn_r
            p,r =  get_prec_rec(tp=tp_r, fp=fp_r, fn=fn_r)
            # print(tokblue('1 p'), p)
            named_p += p
            named_r += r

        # per sentence 'ontology' metrics
        # if True or len(gold_nominal):
        #     nominal_sent_count += 1
        #     tp_o = len(classes_correct)
        #     fp_o = len(pred_nominal) - tp_o
        #     fn_o = len(gold_nominal) - tp_o
        #     TP_O += tp_o
        #     FP_O += fp_o
        #     FN_O += fn_o
        #     p,r =  get_prec_rec(tp=tp_o, fp=fp_o, fn=fn_o)
        #     print(tokblue('2 p'), p)

        #     nominal_p += p
        #     nominal_r += r
        
        # per sentence combined metrics
        # if True or len(gold_combined):
        #     combined_sent_count += 1
        #     tp_c = len(combined_correct)
        #     fp_c = len(pred_combined) - tp_c
        #     fn_c = len(gold_combined) - tp_c
        #     TP_C += tp_c
        #     FP_C += fp_c
        #     FN_C += fn_c
        #     p,r =  get_prec_rec(tp=tp_c, fp=fp_c, fn=fn_c)
        #     print(tokblue('3 p'), p)

        #     combined_p += p
        #     combined_r += r

       # print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))

        sent_count +=1

        #if sent_count == 1:
        #    break
    if printerror:
        error_df =  pd.DataFrame(data, columns = ['Question','gold','pred'])
        error_df.to_csv('error_old.csv', index=False)
    if printerror:
        s = 0
        for i in range(4):
            for j in range(4):
                s += matrix[i][j]
                matrix[i][j] = matrix[i][j] / lengold
                print(round(matrix[i][j],4), end =' ')
            print()
        print(s/ errortotal)

    #macros
    #print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))
    #print("{} {} {} {}".format(named_sent_count, nominal_sent_count, combined_sent_count, skip_count))
    if named_sent_count == 0:
        named_sent_count += 1
    macro_named_p = named_p/named_sent_count
    macro_named_r = named_r/named_sent_count
    macro_named_f1 = get_f1(p=macro_named_p, r=macro_named_r)
    
    if printerror:
        print(tokgreen('macro_named_p'), macro_named_p)

    if nominal_sent_count == 0:
        nominal_sent_count += 1
    macro_nominal_p = nominal_p/nominal_sent_count
    macro_nominal_r = nominal_r/nominal_sent_count
    macro_nominal_f1 = get_f1(p=macro_nominal_p, r=macro_nominal_r)

    if combined_sent_count == 0:
        combined_sent_count += 1
    macro_combined_p = combined_p/combined_sent_count
    macro_combined_r = combined_r/combined_sent_count
    macro_combined_f1 = get_f1(p=macro_combined_p, r=macro_combined_r)

    
    # micros
    micro_named_p, micro_named_r = get_prec_rec(tp=TP_R, fp=FP_R, fn=FN_R)
    micro_named_f1 = get_f1(p=micro_named_p, r=micro_named_r)

    micro_nominal_p, micro_nominal_r = get_prec_rec(tp=TP_O, fp=FP_O, fn=FN_O)
    micro_nominal_f1 = get_f1(p=micro_nominal_p, r=micro_nominal_r)

    micro_combined_p, micro_combined_r = get_prec_rec(tp=TP_C, fp=FP_C, fn=FN_C)
    micro_combined_f1 = get_f1(p=micro_combined_p, r=micro_combined_r)

    nominal = {'precision': round(macro_nominal_p,4), 'recall': round(macro_nominal_r,4), 'f1': round(macro_nominal_f1,4)}
    named = {'precision': round(macro_named_p,4), 'recall': round(macro_named_r,4), 'f1': round(macro_named_f1,4)}
    combined = {'precision': round(macro_combined_p,4), 'recall': round(macro_combined_r,4), 'f1': round(macro_combined_f1,4)}
    macro = {'nominal':nominal, 'named': named, 'combined': combined}

    nominal = {'precision': round(micro_nominal_p,4), 'recall': round(micro_nominal_r,4), 'f1': round(micro_nominal_f1,4)}
    named = {'precision': round(micro_named_p,4), 'recall': round(micro_named_r,4), 'f1': round(micro_named_f1,4)}
    combined = {'precision': round(micro_combined_p,4), 'recall': round(micro_combined_r,4), 'f1': round(micro_combined_f1,4)}
    
    micro = {'nominal':nominal, 'named': named, 'combined': combined}

    metrics['macro'] = macro
    metrics['micro'] = micro
    #print(metrics)
    print('Skip Que Count',skip_count)
    return metrics


def compute_metrics_top_k(benchmark=None, predictions=None, analysisfile=None, limit=250, is_qald_gt=False,
                          eval='full'):
    TP_R, TP_O, TP_C, FP_R, FP_O, FP_C, FN_R, FN_O, FN_C = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    named_p, named_r, nominal_p, nominal_r, combined_p, combined_r = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sent_count = 0
    metrics = {}

    ont_prefix = 'http://dbpedia.org/ontology/'
    res_prefix = 'http://dbpedia.org/resource/'

    named_sent_count = 0
    nominal_sent_count = 0
    combined_sent_count = 0
    skip_count = 0

    if benchmark is None or predictions is None:
        print("Need both benchmark and predicitons to compute metrics")
        return metrics

    result_analysis = pd.DataFrame(columns=['Question', 'Recall_type', 'Recall_val', 'Ground Truth', 'Prediction'],
                                   index=range(0, limit))
    result_analysis_count = 0

    for index, gold_row in benchmark.iterrows():
        if eval == 'dev1' and (sent_count < 250 or sent_count >= 350):
            sent_count += 1
            continue
        if eval == 'dev2' and (sent_count < 350):
            sent_count += 1
            continue

        # if sent_count == 215:
        #    sent_count += 1
        #    skip_count += 1
        #    print(index)
        #    print(sent_count)
        #    continue

        # get nominal and named entities predictions
        if index not in predictions.index:
            skip_count += 1
            #             print(index)
            #             print(sent_count)
            continue

        # (Almaden's groundtruth)
        # get gold nominal and named entities
        if not is_qald_gt:
            gold_entities = gold_row['Entities'].replace('[', '').replace(']', '').replace('\'', '')
            gold_entities = list(gold_entities.split(','))
            gold_nominal = [p.replace(ont_prefix, '').strip() for p in gold_entities if 'ontology' in p]
            gold_named = [p.replace(res_prefix, '').strip() for p in gold_entities if 'resource' in p]
        else:
            # (QALD SPARQL's groundtruth)
            gold_entities = ast.literal_eval(gold_row['Entities'])
            gold_classes = ast.literal_eval(gold_row['Classes'])
            gold_named = [p.replace('res:', '').strip() for p in gold_entities]
            gold_nominal = [p.replace('onto:', '').replace('dbo:', '').strip() for p in gold_classes]

        gold_combined = []
        gold_combined.extend(gold_named)
        gold_combined.extend(gold_nominal)

        new_fp_r = 0
        new_fp_o = 0
        new_fp_c = 0

        pred_entities = ast.literal_eval(str(predictions.loc[index]['Entities']))
        pred_nominal = []
        pred_named = []

        #         print(pred_entities)
        for p2 in pred_entities:
            cycle_nominal = []
            cycle_named = []
            cycle_combined = []
            for p3 in p2:
                if 'ontology' in p3[0]:
                    pred_nominal.append(p3[0].replace(ont_prefix, ''))
                    cycle_combined.append(p3[0].replace(ont_prefix, ''))
                elif 'resource' in p3[0]:
                    pred_named.append(p3[0].replace(res_prefix, ''))
                    cycle_combined.append(p3[0].replace(res_prefix, ''))
            '''
            cycle arrays (cycle_nominal, cycle_named, cycle_combined) contain top-5 dbpedia entities for single amr entity 
            each cycle array is checked against gold standard.
            '''
            r_c = new_overlap(gold_combined, cycle_combined)
            if r_c == 1:
                new_fp_c += 1
                if 'resource' in p2[0][0]:
                    new_fp_r += 1
                elif 'ontology' in p2[0][0]:
                    new_fp_o += 1

        pred_combined = []
        pred_combined.extend(pred_named)
        pred_combined.extend(pred_nominal)

        entities_correct, _ = compute_overlap(gold_named, pred_named, debug=False)
        classes_correct, _ = compute_overlap(gold_nominal, pred_nominal, debug=False)
        combined_correct, _ = compute_overlap(gold_combined, pred_combined)
        # print("{} {} {}".format(entities_correct, classes_correct, combined_correct))

        # per sentence 'resource' metrics
        if True or len(gold_named):
            named_sent_count += 1
            tp_r = len(entities_correct)
            # fp_r = len(pred_named) - tp_r
            fn_r = len(gold_named) - tp_r
            TP_R += tp_r
            FP_R += new_fp_r
            FN_R += fn_r
            p, r = get_prec_rec(tp=tp_r, fp=new_fp_r, fn=fn_r)
            named_p += p
            named_r += r

            # if r < 0.1:
            #     print(index, " Named Recall", r);
            #     print("{} {}".format(pred_named, gold_named))

        # per sentence 'ontology' metrics
        if True or len(gold_nominal):
            nominal_sent_count += 1
            tp_o = len(classes_correct)
            # fp_o = len(pred_nominal) - tp_o
            fn_o = len(gold_nominal) - tp_o
            TP_O += tp_o
            FP_O += new_fp_o
            FN_O += fn_o
            p, r = get_prec_rec(tp=tp_o, fp=new_fp_o, fn=fn_o)
            nominal_p += p
            nominal_r += r

            # if r < 0.1:
            #     print(index, " Nominal Recall", r);
            #     print("{} {}".format(pred_nominal, gold_nominal))

        # per sentence combined metrics
        if True or len(gold_combined):
            combined_sent_count += 1
            tp_c = len(combined_correct)
            # fp_c = len(pred_combined) - tp_c
            fn_c = len(gold_combined) - tp_c
            TP_C += tp_c
            FP_C += new_fp_c
            FN_C += fn_c
            p, r = get_prec_rec(tp=tp_c, fp=new_fp_c, fn=fn_c)
            combined_p += p
            combined_r += r

            if r < 0.1:
                # print(index, " Combined Recall", r);
                # print("{} {}".format(pred_combined, gold_combined))
                result_analysis['Question'][result_analysis_count] = index
                result_analysis['Recall_type'][result_analysis_count] = "Combined"
                result_analysis['Recall_val'][result_analysis_count] = r
                gold_ans = []
                if len(gold_entities) > 0:
                    gold_ans.append(gold_entities)
                if len(gold_classes) > 0:
                    gold_ans.append(gold_classes)
                result_analysis['Ground Truth'][result_analysis_count] = gold_ans
                result_analysis['Prediction'][result_analysis_count] = pred_entities
                result_analysis_count += 1

        # print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))

        sent_count += 1

        if sent_count == limit:
            break

    write_analysis = False
    if write_analysis:
        result_analysis.set_index('Question')
        result_analysis.to_csv(analysisfile, sep=',');

    # macros
    # print("{} {} {} {} || {} {} ".format(named_p, named_r, nominal_p, nominal_r, combined_p, combined_r))
    # print("{} {} {} {}".format(named_sent_count, nominal_sent_count, combined_sent_count, skip_count))
    if named_sent_count == 0:
        named_sent_count += 1

    macro_named_p = named_p / named_sent_count
    macro_named_r = named_r / named_sent_count
    macro_named_f1 = get_f1(p=macro_named_p, r=macro_named_r)

    if nominal_sent_count == 0:
        nominal_sent_count += 1
    macro_nominal_p = nominal_p / nominal_sent_count
    macro_nominal_r = nominal_r / nominal_sent_count
    macro_nominal_f1 = get_f1(p=macro_nominal_p, r=macro_nominal_r)

    if combined_sent_count == 0:
        combined_sent_count += 1
    macro_combined_p = combined_p / combined_sent_count
    macro_combined_r = combined_r / combined_sent_count
    macro_combined_f1 = get_f1(p=macro_combined_p, r=macro_combined_r)

    # micros
    micro_named_p, micro_named_r = get_prec_rec(tp=TP_R, fp=FP_R, fn=FN_R)
    micro_named_f1 = get_f1(p=micro_named_p, r=micro_named_r)

    micro_nominal_p, micro_nominal_r = get_prec_rec(tp=TP_O, fp=FP_O, fn=FN_O)
    micro_nominal_f1 = get_f1(p=micro_nominal_p, r=micro_nominal_r)

    micro_combined_p, micro_combined_r = get_prec_rec(tp=TP_C, fp=FP_C, fn=FN_C)
    micro_combined_f1 = get_f1(p=micro_combined_p, r=micro_combined_r)

    nominal = {'precision': round(macro_nominal_p, 4), 'recall': round(macro_nominal_r, 4),
               'f1': round(macro_nominal_f1, 4)}
    named = {'precision': round(macro_named_p, 4), 'recall': round(macro_named_r, 4), 'f1': round(macro_named_f1, 4)}
    combined = {'precision': round(macro_combined_p, 4), 'recall': round(macro_combined_r, 4),
                'f1': round(macro_combined_f1, 4)}
    macro = {'nominal': nominal, 'named': named, 'combined': combined}

    nominal = {'precision': round(micro_nominal_p, 4), 'recall': round(micro_nominal_r, 4),
               'f1': round(micro_nominal_f1, 4)}
    named = {'precision': round(micro_named_p, 4), 'recall': round(micro_named_r, 4), 'f1': round(micro_named_f1, 4)}
    combined = {'precision': round(micro_combined_p, 4), 'recall': round(micro_combined_r, 4),
                'f1': round(micro_combined_f1, 4)}

    micro = {'nominal': nominal, 'named': named, 'combined': combined}

    metrics['macro'] = macro
    metrics['micro'] = micro
    # print(metrics)

    return metrics


if __name__ == '__main__':
        
    parser = ArgumentParser()
    parser.add_argument("--gold", help="benchmark file")
    parser.add_argument("--pred", help="predictions file")
    parser.add_argument("--gtt", help="ground truth type")
    parser.add_argument("--eval", help="dev1, dev2, full")

    opts = parser.parse_args()


    # read gold file
    if opts.gtt == 'el':
        benchmark = pd.read_csv(opts.gold)
        benchmark = benchmark.set_index('Question')
        benchmark = benchmark.replace(np.nan, '', regex=True)
        benchmark['Entities'] = benchmark['Entities'].astype(object)
        is_qald_gt = False
    elif opts.gtt == 'qald':
        benchmark = pd.read_csv(opts.gold)
        benchmark = benchmark.set_index('Question')
        benchmark = benchmark.replace(np.nan, '', regex=True)
        benchmark['Entities'] = benchmark['Entities'].astype(object)
        #is_qald_gt = False
        #benchmark = pickle.load(open(opts.gold,'rb'))
        #benchmark = benchmark.set_index('Question')
        #benchmark = benchmark.rename(columns={"GT Classes": "Classes", "GT ents": "Entities"})
        is_qald_gt = True



    # read predictions file
    predictions = pd.read_csv(opts.pred)
    predictions = predictions.set_index('Question')
    predictions['Entities'] = predictions['Entities']
    predictions['Classes'] = predictions['Classes']

    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt, eval=opts.eval)

    analysis_files_path = opts.pred.replace('.csv', '')
    analysis_files_path += '_analysis.csv'
    # metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions, analysisfile=analysis_files_path, limit=414, is_qald_gt=is_qald_gt, eval=opts.eval)
    print(metrics)
