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

random.seed(103)

device = torch.device("cpu")

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
# parser.add_argument("--gmt_data", type=str, default="./data/train_sorted.csv", help="train csv")
parser.add_argument("--train_data", type=str, default="./data/train_sorted.csv", help="train csv")
parser.add_argument("--test_data", type=str, default="./data/test.csv", help="test csv")
parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
parser.add_argument("--output_file_name", type=str, default="output/purename_nway_alpha09.txt", help="checkpoint path")
parser.add_argument("--model_name", type=str, default="purename", help="which model we choose")
# args for dividing the corpus
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=100, help='training epochs for LNN')
parser.add_argument("--use_binary", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument('--margin', type=float, default=0.15, help='margin for MarginRankingLoss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument("-f")
args = parser.parse_args()

from RuleLNN_nway_sigmoid_vec import *


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

    # generate the csv
    if mode == 'test':
        df_missing = pd.read_csv("data/missing.csv", header=None)
        df_missing.columns = ['Unnamed:0', 'Question', 'Entities', 'Classes']
        df_missing = df_missing[['Question', 'Entities', 'Classes']]
        df_output = df_output[['Question', 'Entities', 'Classes']]
        df_output = pd.concat([df_output, df_missing], ignore_index=True)
        print("df_output", df_output.shape)

    # gold
    benchmark = pd.read_csv('../../../data/gt_sparql.csv')
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


# def evaluate(eval_model, x_eval, y_eval, m_labels_eval, ques_eval, loss_fn):
#     """evaluate a model on validation data"""
#     eval_model.eval()
#     with torch.no_grad():
#         val_pred = eval_model(x_eval, m_labels_eval)
#         loss = loss_fn(val_pred, y_eval)
#         # print("val loss is {}".format(loss))
#         prec, recall, f1, _ = get_qald_metrics(val_pred, m_labels_eval, ques_eval)
#         # print("Val F1 is {}".format(f1))
#
#     return loss, f1, val_pred


def evaluate_ranking_loss(eval_model, x_, y_, m_labels_, ques_, loss_fn, mode='val'):
    """evaluate a model on validation data"""
    eval_model.eval()

    dataset_ = TensorDataset(x_, y_)
    loader = DataLoader(dataset_, sampler=torch.utils.data.SequentialSampler(dataset_), batch_size=32, shuffle=False)  # always False
    total_loss = 0.0

    with torch.no_grad():
        pred_ = eval_model(x_, m_labels_)

        batch_num = 0
        for xb, yb in loader:
            yhat = eval_model(xb, yb)
            yb = yb.reshape(-1, 1)
            loss = loss_fn(yhat, yb)
            total_loss += loss.item() * (yb.shape[0])
            batch_num += 1
        avg_loss = total_loss / batch_num
        prec, recall, f1, _ = get_qald_metrics(pred_, m_labels_, ques_, mode=mode)  # train and val both use 'val' mode

    return avg_loss, f1, pred_


def train(model, train_data, val_data, test_data, checkpoint_name, num_epochs, margin, learning_rate):
    """train model and tune on validation set"""

    # unwrapping the data
    x_train, y_train, m_labels_train, ques_train = train_data
    x_val, y_val, m_labels_val, ques_val = val_data
    x_test, y_test, m_labels_test, ques_test = test_data

    # inialize the loss function and optimizer
    loss_fn = nn.BCELoss()  # MSELoss(), did not work neither
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_f1, best_val_loss = 0, 100000
    best_train_val_loss = 1000000
    batch_size = 32

    #     stats before training
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
        for xb, yb in loader:
            idx += 1
            model.train()  # set train to true
            optimizer.zero_grad()
            yhat = model(xb, yb)
            yb = yb.reshape(-1, 1)

            loss = loss_fn(yhat, yb)
            if idx == 212:
                print('loss', idx, loss)

            total_loss += loss.item() * (yb.shape[0])
            loss.backward()
            optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, 'param -- data', param.data, 'grad -- ', param.grad)
        # show status after each epoch

        avg_loss = total_loss / idx
        #         train_loss, train_f1, train_pred = evaluate(model, x_train, y_train, m_labels_train, ques_train, loss_fn)
        print("Epoch " + str(epoch) + ": avg train loss -- " + str(avg_loss))
        #         print("Train -- loss is {}; F1 is {}".format(train_loss, train_f1))
        val_loss, val_f1, val_pred = evaluate_ranking_loss(model, x_val, y_val, m_labels_val, ques_val, loss_fn)

        if val_f1 >= best_val_f1:
            # if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            # best_train_val_loss = val_loss + avg_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), checkpoint_name)
        print("Val --  best loss is {}; F1 is {}".format(best_val_loss, best_val_f1))

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
    if model_name == "purename":
        return PureNameLNN(alpha, -1, False)
    elif model_name == "context":
        return ContextLNN(alpha, -1, False)
    elif model_name == "complex":
        return ComplexRuleLNN(alpha, -1, False)
    elif model_name == "lr":
        return LogitsRegression()
    else:
        print("WRONG name input")
        return None


def main():
    # read df and split into train/val
    df_gmt = pd.read_csv(args.train_data)
    df_train_val = pd.read_csv(args.train_data)
    shuffled_question_list = list(df_gmt.Question.unique())
    random.shuffle(shuffled_question_list)
    train_val_split_idx = int(len(shuffled_question_list) * 0.8)
    train_ques_set = shuffled_question_list[:train_val_split_idx]
    val_ques_set = shuffled_question_list[train_val_split_idx:]
    df_train = df_gmt[df_gmt.Question.isin(train_ques_set)]
    df_val = df_train_val[df_train_val.Question.isin(val_ques_set)]
    df_test = pd.read_csv(args.test_data)

    # train
    features_train = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_train.Features.values])
    x_train = torch.from_numpy(features_train).float()
    y_train = torch.from_numpy(df_train.Label.values).float().reshape(-1, 1)
    m_labels_train = df_train.Mention_label.values
    ques_train = df_train.Question.values

    # val
    features_val = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_val.Features.values])
    x_val = torch.from_numpy(features_val).float()
    y_val = torch.from_numpy(df_val.Label.values).float().reshape(-1, 1)
    m_labels_val = df_val.Mention_label.values
    ques_val = df_val.Question.values

    # test
    features_test = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_test.Features.values])
    x_test = torch.from_numpy(features_test).float()
    y_test = torch.from_numpy(df_test.Label.values).float().reshape(-1, 1)
    m_labels_test = df_test.Mention_label.values
    ques_test = df_test.Question.values

    # train model and evaluate
    model = pick_model(args.model_name, args.alpha)
    model = model.to(device)

    # move to gpu
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print(model)

    print("model: ", args.model_name, args.alpha)
    print(model(x_train, m_labels_train))
    print(x_train.shape, x_val.shape)

    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_val sum", sum(y_val), sum(y_val) / len(y_val))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    # aggregate the data into train, val, and test
    train_data = (x_train, y_train, m_labels_train, ques_train)
    print("train:", x_train.shape, y_train.shape, m_labels_train.shape, ques_train.shape)
    val_data = (x_val, y_val, m_labels_val, ques_val)
    print("val:", x_val.shape, y_val.shape, m_labels_val.shape, ques_val.shape)
    test_data = (x_test, y_test, m_labels_test, ques_test)
    print("test:", x_test.shape, y_test.shape, m_labels_test.shape, ques_test.shape)

    # check class distribution
    print("y_train sum", sum(y_train), sum(y_train) / len(y_train))
    print("y_val sum", sum(y_val), sum(y_val) / len(y_val))
    print("y_test sum", sum(y_test), sum(y_test) / len(y_test))

    train(model, train_data, val_data, test_data, args.checkpoint_name, args.num_epoch, args.margin, args.learning_rate)
    test_pred, best_scores = test(x_test, m_labels_test, ques_test, args.alpha, args.checkpoint_name, args.model_name,
                                  args.output_file_name)
    with open("output_w_spacy.txt", 'a') as f:
        f.write("model={}; use_binary={}; alpha={}; p={}; r={}; f1={}; lr={}; margin={}\n".format(args.model_name,
                                                                                                  args.use_binary,
                                                                                                  args.alpha,
                                                                                                  best_scores[
                                                                                                      'precision'],
                                                                                                  best_scores['recall'],
                                                                                                  best_scores['f1'],
                                                                                                  args.learning_rate,
                                                                                                  args.margin))
        print("model={}; use_binary={}; alpha={}; p={}; r={}; f1={}\n".format(args.model_name, args.use_binary,
                                                                              args.alpha,
                                                                              best_scores['precision'],
                                                                              best_scores['recall'],
                                                                              best_scores['f1']))


if __name__ == '__main__':
    # args.learning_rate = 0.001
    args.num_epoch = 200
    args.model_name = "complex"
    # args.margin = 0.601
    args.alpha = 0.9
    args.learning_rate = 0.001
    for margin in np.linspace(0.6, 0.9, num=31):
        args.margin = round(margin, 2)
        print(args.margin)
        main()
    # args.margin = 0.69
    # args.learning_rate = 0.001
    # args.num_epoch = 100
    # main()
    # learning_rate = 1.0
    # for _ in range(0, 6):
    #     learning_rate = learning_rate / 10
    #     args.learning_rate = learning_rate
    #     print(args)
    #     main()
    #
    # learning_rate = 5.0
    # for _ in range(0, 6):
    #     learning_rate = learning_rate / 10
    #     args.learning_rate = learning_rate
    #     print(args)
    #     main()
