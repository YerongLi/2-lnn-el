"""
export EXP_NAME=LNN_EL_without_refcount
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_plus_refcount
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_blink_candidates_only
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount --use_only_blink_candidates --use_blink \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt


export EXP_NAME=LNN_EL_blink
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_refcount --use_blink \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 30 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt

# temporary
export EXP_NAME=LNN_EL_blink
export MODEL_NAME=complex_pure_ctx
python run_lnn_el.py \
--use_blink \
--experiment_name exp_$EXP_NAME_$MODEL_NAME \
--model_name $MODEL_NAME \
--num_epoch 50 \
--margin 0.601 \
--alpha 0.9 \
--learning_rate 0.001 \
--log_file_name log_$EXP_NAME_$MODEL_NAME.log \
--checkpoint_name checkpoint/best_model_$EXP_NAME_$MODEL_NAME.pt

"""

import argparse
from collections import defaultdict
import sys

sys.path.append('../../../')
from el_evaluation import *
from utils import MyBatchSampler
from torch.utils.data import DataLoader, TensorDataset
from utils import QuestionSampler
import copy
import torch
import random
from sklearn.model_selection import KFold
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
random.seed(103)

device = torch.device("cpu")

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
parser.add_argument("--log_file_name", type=str, default="log.txt", help="log_file_name")
parser.add_argument("--experiment_name", type=str, default="exp_lnn", help="checkpoint path")
parser.add_argument("--model_name", type=str, default="complex_pure_ctx", help="which model we choose")
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=30, help='training epochs for LNN')
parser.add_argument("--use_fixed_threshold", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument("--use_kfolds", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument("--use_refcount", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument("--use_blink", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument("--use_only_blink_candidates", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument('--margin', type=float, default=0.601, help='margin for MarginRankingLoss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument("-f")  # to avoid breaking jupyter notebook
args = parser.parse_args()

print("- Refcount + BLINK candidates + BLINK features...")
from models.LNN_EL_blink3 import *

# if not args.use_refcount:
#     print("- Refcount + DBpedia")
#     from models.LNN_EL_without_refcount import *
# elif args.use_refcount and not args.use_only_blink_candidates and not args.use_blink:
#     print("+ Refcount + DBpedia")
#     from models.LNN_EL import *
# elif args.use_blink and args.use_only_blink_candidates:
#     print("+ Refcount + BLINK candidates...")
#     from models.LNN_EL import *
# elif args.use_blink and not args.use_only_blink_candidates:
#     print("+ Refcount + BLINK candidates + BLINK features...")
#     from models.LNN_EL_blink3 import *

# if args.use_blink:
#     print("Containing BLINK features with BLINK candidates...")
#     from models.LNN_EL_blink import *
# else:
#     print("Without BLINK features but with BLINK candidates...")
#     from models.LNN_EL import *


# use the dataset
# if args.use_refcount:
#     print("Containing use_refcount features...")
#     from models.LNN_EL import *
# else:
#     print("Without use_refcount features...")
#     from models.LNN_EL_without_refcount import *


def get_qald_metrics(pred_, m_labels_, ques_, mode='val'):
    """pred_ are 0/1 s after applying a threshold"""
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
    for key, preds_list_mentions in question_rows_map.items():
        if len(preds_list_mentions) > 1:
            rows.append([key, []])
            for preds_set in preds_list_mentions:
                sorted_values = sorted(list(preds_set), key=lambda x: x[1], reverse=True)[:5]
                rows[-1][1].append(sorted_values)
        elif len(preds_list_mentions) == 1:
            sorted_values = sorted(list(preds_list_mentions[0]), key=lambda x: x[1], reverse=True)[:5]
            rows.append([key, [sorted_values]])
        else:
            rows.append([key, []])

    df_output = pd.DataFrame(rows, columns=['Question', 'Entities'])
    df_output['Classes'] = str([])

    # gold
    benchmark = pd.read_csv('../../../data/lcquad/lcquad_gt.csv')
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

    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt,
                              eval='full')

    scores = metrics['macro']['named']
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output


def evaluate_ranking_loss(eval_model, x_, y_, m_labels_, ques_, loss_fn, mode='val'):
    """evaluate a model on validation data"""
    eval_model.eval()

    dataset_ = TensorDataset(x_, y_)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(y_))), y_, False)
    loader = DataLoader(dataset_, batch_sampler=question_sampler, shuffle=False)
    total_loss = 0.0

    with torch.no_grad():
        pred_ = eval_model(x_, m_labels_)

        batch_num = 0
        for xb, yb in loader:
            if yb.shape[0] == 1:
                # print(xb, yb)
                continue
            yhat = eval_model(xb, yb)
            yb = yb.reshape(-1, 1)
            yhat_pos = yhat[-1].repeat((len(yhat) - 1), 1)
            yhat_neg = yhat[:-1]
            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            total_loss += loss.item() * (yb.shape[0] - 1)
            batch_num += 1
        avg_loss = total_loss / batch_num
        prec, recall, f1, _ = get_qald_metrics(pred_, m_labels_, ques_, mode=mode)  # train and val both use 'val' mode

    return avg_loss, f1, pred_


def train(model, train_data, val_data, test_data, checkpoint_name, num_epochs, margin, learning_rate):
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
    best_val_f1, best_val_loss = 0, 100000
    best_train_val_loss = 1000000
    # batch_size = 32

    # stats before training
    print("=========BEFORE TRAINING============")
    train_loss, train_f1, train_pred = evaluate_ranking_loss(model, x_train, y_train, m_labels_train, ques_train,
                                                             loss_fn)
    print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
    val_loss, val_f1, val_pred = evaluate_ranking_loss(model, x_val, y_val, m_labels_val, ques_val, loss_fn)
    print("Val --  loss is {}; F1 is {}".format(val_loss, val_f1))
    test_loss, test_f1, test_pred = evaluate_ranking_loss(model, x_test, y_test, m_labels_test, ques_test, loss_fn,
                                                          mode='test')
    print("Test -- loss is {}; F1 is {}".format(test_loss, test_f1))

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
                # print(prev_xb)
                # print(xb)
                prev_xb[-1, :] = xb
                xb = prev_xb
                # print(xb)
            yhat = model(xb, yb)
            yb = yb.reshape(-1, 1)
            yhat_pos = yhat[-1].repeat((len(yhat) - 1), 1)
            yhat_neg = yhat[:-1]  # torch.mean(yhat[:-1], dim=0)

            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            if idx == 212:
                print('loss', idx, loss)

            total_loss += loss.item() * (yb.shape[0] - 1)
            loss.backward()
            optimizer.step()
            prev_xb, prev_yb = xb, yb

        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, 'param -- data', param.data, 'grad -- ', param.grad)

        # show status after each epoch
        avg_loss = total_loss / idx
        #         train_loss, train_f1, train_pred = evaluate(model, x_train, y_train, m_labels_train, ques_train, loss_fn)
        print("Epoch " + str(epoch) + ": avg train loss -- " + str(avg_loss))
        #         print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
        val_loss, val_f1, val_pred = evaluate_ranking_loss(model, x_val, y_val, m_labels_val, ques_val, loss_fn)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        if val_f1 >= best_val_f1:
            # if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            # best_train_val_loss = val_loss + avg_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), checkpoint_name)
        print("Val --  best loss (so far) is {}; F1 is {}".format(best_val_loss, best_val_f1))

        test_loss, test_f1, test_pred = evaluate_ranking_loss(model, x_test, y_test, m_labels_test, ques_test, loss_fn,
                                                              mode='test')
        print("Current Test -- loss is {}; F1 is {}".format(test_loss, test_f1))

    # show stats after training
    print("=========AFTER TRAINING============")
    train_loss, train_f1, train_pred = evaluate_ranking_loss(best_model, x_train, y_train, m_labels_train, ques_train,
                                                             loss_fn)
    print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
    val_loss, val_f1, val_pred = evaluate_ranking_loss(best_model, x_val, y_val, m_labels_val, ques_val, loss_fn)
    print("Val --  loss is {}; F1 is {}".format(val_loss, val_f1))
    test_loss, test_f1, test_pred = evaluate_ranking_loss(best_model, x_test, y_test, m_labels_test, ques_test, loss_fn,
                                                          mode='test')
    print("Test -- loss is {}; F1 is {}".format(test_loss, test_f1))

    # finished logging
    writer.close()

    return


def test(x_test, m_labels_test, ques_test, alpha, checkpoint_name, model_name, output_file_name):
    """make predictions on test set"""
    bestModel = pick_model(model_name, alpha)
    bestModel.load_state_dict(torch.load(checkpoint_name))
    bestModel.eval()
    best_scores = {}

    with torch.no_grad():
        test_pred = bestModel(x_test, m_labels_test)
        prec, recall, f1, df_output = get_qald_metrics(test_pred, m_labels_test, ques_test, mode='test')
        df_output.to_csv(output_file_name)
        print("Test -- f1 is {} ".format(f1))
        print("Test -- prec, recall, f1", prec, recall, f1)
        best_scores['precision'] = prec
        best_scores['recall'] = recall
        best_scores['f1'] = f1

    # for name, mod in bestModel.named_modules():
    #     if type(mod) == nn.ModuleList:
    #         for name1, mod1 in mod.named_modules():
    #             if 'cdd' not in name1 and 'AND' not in name1:
    #                 if 'batch' in name1.lower():
    #                     continue
    #                 elif 'or_max' in name1.lower():
    #                     continue
    #                 elif 'and' in name1.lower():
    #                     print(name1, mod1.cdd())
    #                 elif 'or' in name1.lower():
    #                     print(name1, mod1.AND.cdd())
    #     else:
    #         if 'cdd' not in name and 'AND' not in name:
    #             if 'batch' in name.lower():
    #                 continue
    #             elif 'or_max' in name.lower():
    #                 continue
    #             elif 'and' in name.lower():
    #                 print(name, mod.cdd())
    #             elif 'or' in name.lower():
    #                 print(name, mod.AND.cdd())
    return test_pred, best_scores


def pick_model(model_name, alpha):
    """returns the model object"""
    if model_name == "purename":
        return PureNameLNN(alpha, -1, False)
    elif model_name == "context":
        return ContextLNN(alpha, -1, False)
    elif model_name == "type":
        return TypeLNN(alpha, -1, False)
    elif model_name == "complex_pure_ctx":
        print("===ComplexRuleWithoutTypeLNN===")
        return ComplexRuleWithoutTypeLNN(alpha, -1, False)
    elif model_name == "complex_pure_ctx_type":
        return ComplexRuleWithTypeLNN(alpha, -1, False)
    elif model_name == "lr":
        return LogitsRegression()
    else:
        print("WRONG name input")
        return None


def main_predefined_split():
    """Runs an experiment on pre-defined train/test splits"""

    average_performance = []
    fold_num = 'predefined'
    output_file_folder = "output/{}".format(args.experiment_name)
    output_file_name = "{}/lnnel_{}.csv".format(output_file_folder, fold_num)
    Path(output_file_folder).mkdir(parents=True, exist_ok=True)
    args.output_file_name = output_file_name

    if args.use_blink:
        df_train = pd.read_csv("./data/lcquad/blink/lcquad_train_sorted.csv")
        df_test = pd.read_csv("./data/lcquad/blink/lcquad_test_sorted.csv")
    else:
        df_train = pd.read_csv("./data/lcquad/dbpedia/lcquad_train_sorted.csv")
        df_test = pd.read_csv("./data/lcquad/dbpedia/lcquad_test_sorted.csv")

    # filter out the questions with single positive or many negatives in trianing set
    filtered_question_mentions = []
    for qm in df_train.QuestionMention.unique():
        df_ = df_train[df_train.QuestionMention == qm]
        if df_.Label.sum() == 0:
            filtered_question_mentions.append(qm)
        if df_.Label.sum() == 1 and df_.shape[0] == 1:
            filtered_question_mentions.append(qm)
    #             print(df_.Label.values)
    df_train_split_filtered = df_train[~df_train.QuestionMention.isin(filtered_question_mentions)]
    df_train_split_filtered = df_train_split_filtered.sort_values(by=['QuestionMention', 'Label'])
    df_train = df_train_split_filtered

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

    # train model and evaluate
    model = pick_model(args.model_name, args.alpha)
    model = model.to(device)

    # move to gpu
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print(model)

    print("model: ", args.model_name, args.alpha)
    print(model(x_train, m_labels_train))

    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    # aggregate the data into train, val, and test
    train_data = (x_train, y_train, m_labels_train, ques_train)
    print("train:", x_train.shape, y_train.shape, m_labels_train.shape, ques_train.shape)
    test_data = (x_test, y_test, m_labels_test, ques_test)
    print("test:", x_test.shape, y_test.shape, m_labels_test.shape, ques_test.shape)

    # check class distribution
    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    train(model, train_data, test_data, test_data, args.checkpoint_name, args.num_epoch, args.margin,
          args.learning_rate)
    test_pred, best_scores = test(x_test, m_labels_test, ques_test, args.alpha, args.checkpoint_name,
                                  args.model_name,
                                  args.output_file_name)
    with open(args.log_file_name, 'a') as f:
        f.write(
            "model={}; use_fixed_threshold={}; alpha={}; p={}; r={}; f1={}; lr={}; margin={}\n".format(
                args.model_name,
                args.use_fixed_threshold,
                args.alpha,
                best_scores[
                    'precision'],
                best_scores[
                    'recall'],
                best_scores['f1'],
                args.learning_rate,
                args.margin))
        print("model={}; use_fixed_threshold={}; alpha={}; p={}; r={}; f1={}\n".format(args.model_name,
                                                                                       args.use_fixed_threshold,
                                                                                       args.alpha,
                                                                                       best_scores['precision'],
                                                                                       best_scores['recall'],
                                                                                       best_scores['f1']))
    average_performance.append([best_scores['precision'], best_scores['recall'], best_scores['f1']])

    average_performance = np.array(average_performance)
    print("Avg performance is prec - rec - f1: ", average_performance.mean(0))


def main_cross_validation():
    average_performance = []
    for fold_num in range(1, 6):
        output_file_folder = "output/{}".format(args.experiment_name)
        output_file_name = "{}/lnnel_kfolds_{}.csv".format(output_file_folder, fold_num)
        Path(output_file_folder).mkdir(parents=True, exist_ok=True)
        args.output_file_name = output_file_name

        if args.use_blink:
            df_train = pd.read_csv("./data/lcquad/blink/lcquad-5-folds/split{}/train_fold.csv".format(fold_num))
            df_test = pd.read_csv("./data/lcquad/blink/lcquad-5-folds/split{}/test_fold.csv".format(fold_num))
        else:
            df_train = pd.read_csv("./data/lcquad/dbpedia/lcquad-5-folds/split{}/train_fold.csv".format(fold_num))
            df_test = pd.read_csv("./data/lcquad/dbpedia/lcquad-5-folds/split{}/test_fold.csv".format(fold_num))

        # filter out the questions with single positive or many negatives in trianing set
        filtered_question_mentions = []
        for qm in df_train.QuestionMention.unique():
            df_ = df_train[df_train.QuestionMention == qm]
            if df_.Label.sum() == 0:
                filtered_question_mentions.append(qm)
            if df_.Label.sum() == 1 and df_.shape[0] == 1:
                filtered_question_mentions.append(qm)
        #             print(df_.Label.values)
        df_train_split_filtered = df_train[~df_train.QuestionMention.isin(filtered_question_mentions)]
        df_train_split_filtered = df_train_split_filtered.sort_values(by=['QuestionMention', 'Label'])
        df_train = df_train_split_filtered

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

        # train model and evaluate
        model = pick_model(args.model_name, args.alpha)
        model = model.to(device)

        # move to gpu
        x_train, y_train = x_train.to(device), y_train.to(device)
        x_test, y_test = x_test.to(device), y_test.to(device)

        print(model)

        print("model: ", args.model_name, args.alpha)
        print(model(x_train, m_labels_train))

        print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
        print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

        # aggregate the data into train, val, and test
        train_data = (x_train, y_train, m_labels_train, ques_train)
        print("train:", x_train.shape, y_train.shape, m_labels_train.shape, ques_train.shape)
        test_data = (x_test, y_test, m_labels_test, ques_test)
        print("test:", x_test.shape, y_test.shape, m_labels_test.shape, ques_test.shape)

        # check class distribution
        print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
        print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

        train(model, train_data, test_data, test_data, args.checkpoint_name, args.num_epoch, args.margin,
              args.learning_rate)
        test_pred, best_scores = test(x_test, m_labels_test, ques_test, args.alpha, args.checkpoint_name,
                                      args.model_name,
                                      args.output_file_name)
        with open(args.log_file_name, 'a') as f:
            f.write(
                "model={}; use_fixed_threshold={}; alpha={}; p={}; r={}; f1={}; lr={}; margin={}\n".format(args.model_name,
                                                                                                  args.use_fixed_threshold,
                                                                                                  args.alpha,
                                                                                                  best_scores[
                                                                                                      'precision'],
                                                                                                  best_scores[
                                                                                                      'recall'],
                                                                                                  best_scores['f1'],
                                                                                                  args.learning_rate,
                                                                                                  args.margin))
            print("model={}; use_fixed_threshold={}; alpha={}; p={}; r={}; f1={}\n".format(args.model_name, args.use_fixed_threshold,
                                                                                  args.alpha,
                                                                                  best_scores['precision'],
                                                                                  best_scores['recall'],
                                                                                  best_scores['f1']))
        average_performance.append([ best_scores['precision'], best_scores['recall'], best_scores['f1']])
    average_performance = np.array(average_performance)
    print("Avg performance is prec - rec - f1: ", average_performance.mean(0))
    with open(args.log_file_name, 'a') as f:
        f.write("Average performance is prec {} - rec {} - f1 {} \n".format(average_performance.mean(0)[0], average_performance.mean(0)[1], average_performance.mean(0)[2]))


if __name__ == '__main__':
    if args.use_kfolds:
        print("Running on five fold cross validation")
        main_cross_validation()
    else:
        print("Running on predefined train/test split")
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
