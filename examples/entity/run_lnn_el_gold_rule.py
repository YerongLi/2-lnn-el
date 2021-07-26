"""
[scores['jw'], scores['jacc'], scores['lev'], scores['spacy'],
scores['in'], scores['pr'], scores['smith_waterman'] ,
normalized_ref_scores[ref_idx], normalized_ctx_scores[ctx_idx],
has_class, blink_score, rel_contex_score]

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name pure_ctx_type \
--experiment_name exp_lnn_lcquad_pure_ctx_type \
--num_epoch 30 \
--learning_rate 0.0001 \

--margin 0.75


python run_lnn_el_gold.py \
--dataset_name qald9 \
--model_name pure_ctx_type \
--experiment_name exp_lnn_qald9_pure_ctx_type \
--num_epoch 30 \
--learning_rate 0.005

python run_lnn_el_gold.py \
--dataset_name webqsp \
--model_name pure_ctx_type \
--experiment_name exp_lnn_webqsp_pure_ctx_type \
--num_epoch 30 \
--learning_rate 0.0001


python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name pure_ctx \
--experiment_name exp_lnn_lcquad_pure_ctx \
--num_epoch 30 \
--learning_rate 0.0001

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name pure_type \
--experiment_name exp_lnn_lcquad_pure_type \
--num_epoch 30 \
--learning_rate 0.0001

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name ctx_type \
--experiment_name exp_lnn_lcquad_ctx_type \
--num_epoch 30 \
--learning_rate 0.0001

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name pure \
--experiment_name exp_lnn_lcquad_pure \
--num_epoch 30 \
--learning_rate 0.0001

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name ctx \
--experiment_name exp_lnn_lcquad_ctx \
--num_epoch 30 \
--learning_rate 0.0001

python run_lnn_el_gold.py \
--dataset_name lcquad \
--model_name type \
--experiment_name exp_lnn_lcquad_type \
--num_epoch 30 \
--learning_rate 0.0001
"""

import argparse
from collections import defaultdict
import sys

sys.path.append('../../../')
from utils import MyBatchSampler
from torch.utils.data import DataLoader, TensorDataset
from utils import QuestionSampler
import copy, os
import torch
import random, math
from sklearn.model_selection import KFold
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
random.seed(103)

device = torch.device("cpu")

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
# parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
parser.add_argument("--dataset_name", type=str, default="lcquad", help="log_file_name")
parser.add_argument("--experiment_name", type=str, default="exp_lnn", help="checkpoint path")
parser.add_argument("--model_name", type=str, default="complex_pure_ctx", help="which model we choose")
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=30, help='training epochs for LNN')
parser.add_argument("--topk", type=int, default=5)
# parser.add_argument("--use_fixed_threshold", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_kfolds", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_refcount", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_blink", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_only_blink_candidates", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument('--margin', type=float, default=0.601, help='margin for MarginRankingLoss')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument("-f")  # to avoid breaking jupyter notebook
args = parser.parse_args()

from models.LNN_EL_gold_db import *
from el_evaluation import *
# from el_evaluation_redirect import *


def get_qald_metrics(pred_, m_labels_, ques_, gold_file_name, topk=5):
    """pred_ are 0/1 s after applying a threshold"""

    # try:
    #     n, l, r = df_.shape[0], 0, 0
    #     data = []
    #     while r < n:
    #         while r < n - 1 and df_.iloc[l]['QuestionMention'] == df_.iloc[r + 1]['QuestionMention']:
    #             r+= 1
    #         batch = df_.iloc[l : r + 1]
    #         gold_pairs = batch[batch.Label.eq(1)]['Mention_label'].values
    #         # assert(len(gold_pairs) == 1)
    #         gt = gold_pairs[0].split(';')[1].replace(' ', '_')
    #         j = torch.argmax(pred_[l:r + 1]).numpy()
    #         # print(j)
    #         if j != r - l:
    #             data.append([df_.iloc[l]['QuestionMention'],
    #             f'https://dbpedia.org/page/{gt}', 
    #             f"https://dbpedia.org/page{batch.iloc[j]['Mention_label'].split(';')[1].replace(' ', '_')}"])
    #         l = r + 1
    #         r = l
    #     error_df =  pd.DataFrame(data, columns = ['QuestionMention','gold','pred'])
    #     error_df.to_csv('error.csv', index=False)
    # except:
        # pass
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
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred))
            else:
                question_mention_set.add('-'.join([question, men_entity_mention]))
                question_rows_map[ques_[i]].append(set())
                question_rows_map[ques_[i]][-1].add(
                    ('http://dbpedia.org/resource/{}'.format(men_entity_label), pred))
    # print(question_rows_map.keys())
    for key, preds_list_mentions in question_rows_map.items():
        if len(preds_list_mentions) > 1:
            rows.append([key, []])
            for preds_set in preds_list_mentions:
                sorted_values = sorted(list(preds_set), key=lambda x: x[1], reverse=True)[:topk]
                rows[-1][1].append(sorted_values)
        elif len(preds_list_mentions) == 1:
            sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: x[1], reverse=True)[:topk]
            rows.append([key, [sorted_values]])
        else:
            rows.append([key, []])

    df_output = pd.DataFrame(rows, columns=['Question', 'Entities'])
    df_output['Classes'] = str([])

    # gold
    benchmark = pd.read_csv('data/{}'.format(gold_file_name))
    benchmark = benchmark.set_index('Question')
    benchmark = benchmark.replace(np.nan, '', regex=True)
    benchmark['Entities'] = benchmark['Entities'].astype(object)
    is_qald_gt = True

    # pred
    predictions = df_output
    # print(df_output.shape)
    def rep(q):
        return q.replace(",", "@").replace('"', '#')
    predictions.Question = predictions.Question.apply(rep)
    predictions = predictions.set_index('Question')
    predictions['Entities'] = predictions['Entities']
    predictions['Classes'] = predictions['Classes']

    # compute_metrics_top_k
    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt,
                              eval='full')

    scores = metrics['macro']['named']
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output


def evaluate_ranking_loss(eval_model, x_, y_, m_labels_, ques_, loss_fn, mode='val', is_long= False):
    """evaluate a model on validation data"""
    eval_model.eval()

    dataset_ = TensorDataset(x_, y_)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(y_))), y_, False)
    loader = DataLoader(dataset_, batch_sampler=question_sampler, shuffle=False)
    total_loss = 0.0

    # pred_ = []
    with torch.no_grad():
        pred_ = eval_model(x_, m_labels_)
        batch_num = 0
        for xb, yb in loader:
            if (yb.shape[0] == 1):
                # print(xb, yb)
                # pred_.append(torch.tensor([[1.]]))
                continue
            yhat = eval_model(xb, yb)
            # pred_.append(yhat)
            yb = yb.reshape(-1, 1)
            yhat_pos = yhat[-1].repeat((len(yhat) - 1), 1)
            yhat_neg = yhat[:-1]
            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            total_loss += loss.item() * (yb.shape[0] - 1)
            batch_num += 1
        # pred_ = torch.cat(pred_, 0)
        avg_loss = total_loss / batch_num
        
        if is_long:
            prec, recall, f1, df = compute_long_metrics(pred_, m_labels_, ques_, args.gold_file_name, topk=args.topk)  # train and val both use 'val' mode
        else:
            prec, recall, f1, df = get_qald_metrics(pred_, m_labels_, ques_, args.gold_file_name, topk=args.topk)  # train and val both use 'val' mode
    performance = [prec, recall, f1]
    return avg_loss, performance, pred_, df


def train(model, train_data, val_data, test_data, checkpoint_name, num_epochs, margin, learning_rate,is_long):
    """train model and tune on validation set"""

    # start logging
    writer = SummaryWriter()

    # unwrapping the data
    x_train, y_train, m_labels_train, ques_train = train_data
    x_val, y_val, m_labels_val, ques_val = val_data
    x_test, y_test, m_labels_test, ques_test = test_data

    # initialize the loss function and optimizer
    loss_fn = nn.MarginRankingLoss(margin=margin)  # MSELoss(), did not work neither
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_f1, best_val_loss = 0, math.inf

    # stats before training
    # print("=========BEFORE TRAINING============")
    # train_loss, train_f1, train_pred, _ = evaluate_ranking_loss(model, x_train, y_train, m_labels_train, ques_train,
    #                                                          loss_fn, mode="train")
    # print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
    # val_loss, (val_prec, val_recall, val_f1), val_pred, _  = evaluate_ranking_loss(model, x_val, y_val, m_labels_val, ques_val, loss_fn, mode="val")
    # print("Val --  loss is {}; F1 is {}".format(val_loss, val_f1))
    # test_loss, test_f1, test_pred, _  = evaluate_ranking_loss(model, x_test, y_test, m_labels_test, ques_test, loss_fn,
    #                                                       mode='test')
    # print("Test -- loss is {}; F1 is {}".format(test_loss, test_f1))

    # start training
    print("=========TRAINING============")
    dataset_train = TensorDataset(x_train, y_train)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(y_train))), y_train, False)
    loader = DataLoader(dataset_train, batch_sampler=question_sampler, shuffle=False)
    # loader = DataLoader(dataset_train, sampler=torch.utils.data.SequentialSampler(dataset_train), batch_size=batch_size, shuffle=False)  # always False
    # loader = DataLoader(dataset_train, sampler=torch.utils.data.WeightedRandomSampler(torch.FloatTensor([1, 100]), len(x_train), replacement=True), batch_size=64, shuffle=False)  # always False
    best_model = None

    for epoch in range(num_epochs):
        total_loss = 0.0
        idx = 0

        # prev xb
        prev_xb, prev_yb = None, None
        for xb, yb in loader:
            idx += 1
            model.train()  # set train to true
            optimizer.zero_grad()
            # handle question with multiple positives
            if yb.shape[0] == 1:
                yb = prev_yb
                prev_xb[-1, :] = xb
                xb = prev_xb
            yhat = model(xb, yb)
            yb = yb.reshape(-1, 1)
            yhat_pos = yhat[-1].repeat((len(yhat) - 1), 1)
            yhat_neg = yhat[:-1]  # torch.mean(yhat[:-1], dim=0)
            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            total_loss += loss.item() * (yb.shape[0] - 1)
            loss.backward()
            optimizer.step()
            prev_xb, prev_yb = xb, yb

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, 'param -- data', param.data, 'grad -- ', param.grad)

        # show status after each epoch
        avg_loss = total_loss / idx
        #         train_loss, train_f1, train_pred = evaluate(model, x_train, y_train, m_labels_train, ques_train, loss_fn)
        print("Epoch " + str(epoch) + ": avg train loss -- " + str(avg_loss))
        #         print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
        val_loss, (val_prec, val_rec, val_f1), val_pred, _  = evaluate_ranking_loss(model, x_val, y_val, m_labels_val, ques_val, loss_fn,is_long)
        print("Val loss {}; Prec {}; Recall {}; F1 {}".format(val_loss, val_prec, val_rec, val_f1))
        test_loss, (test_prec, test_rec, test_f1), test_pred, df_output = \
            evaluate_ranking_loss(model, x_test, y_test, m_labels_test, ques_test, loss_fn, mode='test', is_long=is_long)
        print("Test -- loss is {}; Prec {}; Recall {}; F1 {}".format(test_loss, test_prec, test_rec, test_f1))
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        # torch.save(model.state_dict(), checkpoint_name)
        # df_output.to_csv(args.output_file_name)
        print("Best Test -- loss is {}; Prec {}; Recall {}; F1 {}".format(test_loss, test_prec, test_rec, test_f1))
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), checkpoint_name)
            df_output.to_csv(args.output_file_name)
            with open(args.log_file_name, "a") as out:
                log_str = "Best Test -- loss is {}; Prec {}; Recall {}; F1 {}".format(test_loss, test_prec, test_rec, test_f1)
                out.write(log_str)
                out.write("\n")
                print(log_str)

    # finished logging
    writer.close()

    return


def read_data_file(df_, device, split_name="train"):
    """
    read the lcquad data from DBpedia (not combined version)
    """
    features_train = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_.Features.values])
    x_ = torch.from_numpy(features_train).float().to(device)
    y_ = torch.from_numpy(df_.Label.values).float().reshape(-1, 1).to(device)
    m_labels_ = df_.Mention_label.values
    ques_ = df_.Question.values
    print("loading {}".format(split_name))
    return x_, y_, m_labels_, ques_


def main_predefined_split():
    """Runs an experiment on pre-defined train/test splits"""

    # set up output directory and file
    output_file_folder = "output/{}".format(args.experiment_name)
    Path(output_file_folder).mkdir(parents=True, exist_ok=True)
    args.output_file_name = "{}/{}.csv".format(output_file_folder, args.model_name)
    args.checkpoint_name = "{}/{}.pt".format(output_file_folder, args.model_name + "_best_model")
    args.log_file_name = "{}/{}.txt".format(output_file_folder, args.model_name + "_logs")

    if args.dataset_name == "lcquad":
        df_train = pd.read_csv("./data/lcquad/blink_bert_box/full_train_gold.csv")
        df_valid = pd.read_csv("./data/lcquad/blink_bert_box/full_valid_gold.csv")
        df_test = pd.read_csv("./data/lcquad/blink_bert_box/full_test_gold.csv")
        # df_train = pd.read_csv("./data/lcquad/merge_gold_db_blink/train_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # df_valid = pd.read_csv("./data/lcquad/merge_gold_db_blink/valid_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # df_test = pd.read_csv("./data/lcquad/merge_gold_db_blink/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        args.gold_file_name = "lcquad/full_lcquad_gt_5000.csv"
        # df_test = pd.read_csv("./data/qald-9/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "qald/qald_data_gt.csv"
        # df_test = pd.read_csv("./data/webqsp/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "webqsp/webqsp_data_gt.csv"
    elif args.dataset_name == "qald9":
        df_train = pd.read_csv("./data/qald-9/blink_bert_box/train_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        df_valid = pd.read_csv("./data/qald-9/blink_bert_box/valid_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        df_test = pd.read_csv("./data/qald-9/blink_bert_box/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        args.gold_file_name = "qald/qald_data_gt.csv"
        # df_test = pd.read_csv("./data/lcquad/merge_gold_db_blink/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "lcquad/lcquad_gt_5000.csv"
        # df_test = pd.read_csv("./data/webqsp/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "webqsp/webqsp_data_gt.csv"
    elif args.dataset_name == "webqsp":
        df_train = pd.read_csv("./data/webqsp/blink_bert_box/train_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        df_valid = pd.read_csv("./data/webqsp/blink_bert_box/valid_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        df_test = pd.read_csv("./data/webqsp/blink_bert_box/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        args.gold_file_name = "webqsp/webqsp_data_gt.csv"
    
    if args.dataset_name == "aida":
        df_train = pd.read_csv("./data/aida/blink_bert_box/full_train.csv")
        df_valid = pd.read_csv("./data/aida/blink_bert_box/full_testA.csv")
        df_test = pd.read_csv("./data/aida/blink_bert_box/full_testB.csv")
        # df_train = pd.read_csv("./data/lcquad/merge_gold_db_blink/train_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # df_valid = pd.read_csv("./data/lcquad/merge_gold_db_blink/valid_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # df_test = pd.read_csv("./data/lcquad/merge_gold_db_blink/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        args.gold_file_name = "aida/full_aida_gt_5000.csv"
        # df_test = pd.read_csv("./data/qald-9/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "qald/qald_data_gt.csv"
        # df_test = pd.read_csv("./data/webqsp/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
        # args.gold_file_name = "webqsp/webqsp_data_gt.csv"
    print(f'training set {df_train}')
    is_long = args.dataset_name == 'aida'
    # elif (args.dataset_name == 'lcquad_bert'):
    #     """ including bert embedding
    #     python run_lnn_el_gold.py \
    #     --dataset_name lcquad_bert \
    #     --model_name pure_ctx_type_bert\
    #     --experiment_name exp_lnn_lcquad_bert_pure_ctx_type \
    #     --num_epoch 30 \
    #     --learning_rate 0.0001
    #     """
    #     df_train = pd.read_csv("./data/lcquad/bert/train_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_valid = pd.read_csv("./data/lcquad/bert/valid_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_test = pd.read_csv("./data/lcquad/bert/test_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     args.gold_file_name = "lcquad/lcquad_gt_5000.csv"
    # elif (args.dataset_name == 'lcquad_box'):
    #     """
    #     python run_lnn_el_gold.py \
    #     --dataset_name lcquad_box \
    #     --model_name pure_ctx_type_box \
    #     --experiment_name exp_lnn_lcquad_box_pure_ctx_type \
    #     --num_epoch 30 \
    #     --learning_rate 0.0001
    #     """
    #     df_train = pd.read_csv("./data/lcquad/box/train_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_valid = pd.read_csv("./data/lcquad/box/valid_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_test = pd.read_csv("./data/lcquad/box/test_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     args.gold_file_name = "lcquad/lcquad_gt_5000.csv"
    # elif (args.dataset_name == 'qald_bert'):
    #     """
    #     python run_lnn_el_gold.py \
    #     --dataset_name qald_bert \
    #     --model_name pure_ctx_type_bert\
    #     --experiment_name exp_lnn_qald_bert_pure_ctx_type \
    #     --num_epoch 30 \
    #     --learning_rate 0.005
    #     """
    #     df_train = pd.read_csv("./data/qald-9/bert/train_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_valid = pd.read_csv("./data/qald-9/bert/valid_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_test = pd.read_csv("./data/qald-9/bert/test_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     args.gold_file_name = "qald/qald_data_gt.csv"
    # elif (args.dataset_name == 'qald_box'):
    #     """
    #     python run_lnn_el_gold.py \
    #     --dataset_name qald_box \
    #     --model_name pure_ctx_type_box \
    #     --experiment_name exp_lnn_qald_box_pure_ctx_type \
    #     --num_epoch 30 \
    #     --margin 0.75 \
    #     --learning_rate 0.005
    #     """
    #     df_train = pd.read_csv("./data/qald-9/box/train_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_valid = pd.read_csv("./data/qald-9/box/valid_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     df_test = pd.read_csv("./data/qald-9/box/test_gold.csv").sort_values(
    #         by=['QuestionMention', 'Label'])
    #     args.gold_file_name = "qald/qald_data_gt.csv"
    # elif args.dataset_name == "webqsp":
    #     df_train = pd.read_csv("./data/webqsp/train_gold.csv").sort_values(by=['QuestionMention', 'Label'])
    #     df_valid = pd.read_csv("./data/webqsp/valid_gold.csv").sort_values(by=['QuestionMention', 'Label'])
    #     df_test = pd.read_csv("./data/webqsp/test_gold.csv").sort_values(by=['QuestionMention', 'Label'])
    #     args.gold_file_name = "webqsp/webqsp_data_gt.csv"

    train_data = read_data_file(df_train, device, "train")
    valid_data = read_data_file(df_valid, device, "valid")
    test_data = read_data_file(df_test, device, "test")

    # train model and evaluate
    if args.model_name == "pure":
        model = PureNameLNN(args.alpha, -1, False)
    elif args.model_name == "ctx":
        model = ContextLNN(args.alpha, -1, False)
    elif args.model_name == 'type':
        model = TypeLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx":
        model = PureCtxLNN(args.alpha, -1, False)
    elif args.model_name == "pure_type":
        model = PureTypeLNN(args.alpha, -1, False)
    elif args.model_name == "ctx_type":
        model = CtxTypeLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type":
        model = ComplexRuleTypeLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type_bert":
        model = ComplexBertRulesLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type_box":
        model = ComplexBoxRulesLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type_blink":
        model = ComplexBlinkRulesLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type_blink_box":
        model = ComplexBlinkBoxRulesLNN(args.alpha, -1, False)
    elif args.model_name == "pure_ctx_type_blink_box_bert":
        model = ComplexBlinkBoxBertRulesLNN(args.alpha, -1, False)
    model = model.to(device)
    print(f'model type : {type(model)}')
    train(model, train_data, valid_data, test_data, args.checkpoint_name, args.num_epoch, args.margin,
          args.learning_rate, is_long)

    # evaluate
    model.load_state_dict(torch.load(args.checkpoint_name))
    model.eval()
    x_test, y_test, m_labels_test, ques_test = test_data
    pred_ = model(x_test)
    prec, recall, f1, df = get_qald_metrics(pred_, m_labels_test, ques_test, args.gold_file_name, topk=args.topk)  # train and val both use 'val' mode
    print("Load and Test: prec {}; rec {}; f1 {}.".format(prec, recall, f1))
    df.to_csv("{}/prediction_{}".format(output_file_folder, os.path.basename(args.gold_file_name)))


if __name__ == '__main__':
    main_predefined_split()

    # code to train each of the models
    # args.learning_rate = 0.001
    # args.use_refcount = True
    # args.use_blink = True
    # for learning_rate in [0.01, 0.001, 0.0001]:
    #     args.model_name = "complex_pure_ctx"
    #     args.experiment_name = "exp_lnn_1019_{}".format(args.model_name)
    #     args.num_epoch = 30
    #     args.margin = 0.601
    #     args.alpha = 0.9
    #     args.learning_rate = 0.001
    #     args.use_kfolds = True
    #     args.log_file_name = "log_{}.txt".format(args.experiment_name)
    #     args.checkpoint_name = "checkpoint/best_model_{}.pt".format(args.experiment_name)
    #     main_predefined_split()
