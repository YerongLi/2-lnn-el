import argparse
from collections import defaultdict
import sys
sys.path.append('../../../')
from el_evaluation import *

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
parser.add_argument("--train_data", type=str, default="./data/train.csv", help="train csv")
parser.add_argument("--test_data", type=str, default="./data/test.csv", help="test csv")
parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
parser.add_argument("--output_file_name", type=str, default="output/purename_nway_alpha09.txt", help="checkpoint path")
parser.add_argument("--model_name", type=str, default="purename", help="which model we choose")
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=100, help='training epochs for LNN')

args = parser.parse_args()

from RuleEL import *


def get_qald_metrics(val_pred, val_m_labels, ques_val, mode='val'):
    """val_pred are 0/1 s after applying a threshold"""
    rows = []
    question_rows_map = {}
    question_mention_set = set()
    for i, pred in enumerate(val_pred):
        pred = pred.data.tolist()[0]
        question = ques_val[i]
        if question not in question_rows_map:
            question_rows_map[ques_val[i]] = []
        if pred:
            men_entity_label = '_'.join(val_m_labels[i].split(';')[-1].split())
            men_entity_mention = '_'.join(val_m_labels[i].split(';')[0].split())
            if '-'.join([question, men_entity_mention]) in question_mention_set:
                question_rows_map[ques_val[i]][-1].add(('http://dbpedia.org/resource/{}'.format(men_entity_label), pred))
            else:
                question_mention_set.add('-'.join([question, men_entity_mention]))
                question_rows_map[ques_val[i]].append(set())
                question_rows_map[ques_val[i]][-1].add(('http://dbpedia.org/resource/{}'.format(men_entity_label), pred))
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
        # print(df_output.head())

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
    predictions.to_csv("tmp.csv")
    metrics = compute_metrics(benchmark=benchmark, predictions=predictions, limit=410, is_qald_gt=is_qald_gt,
                              eval='full')

    scores = metrics['macro']['named']
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output


def test(x_test, m_labels_test, ques_test, best_tuned_threshold, alpha, checkpoint_name, model_name, output_file_name):
    """make predictions on test set"""
    bestModel = pick_model(model_name, alpha)
    # bestModel.load_state_dict(torch.load(checkpoint_name))
    bestModel.eval()
    best_scores = {}

    with torch.no_grad():
        test_pred = bestModel(x_test, m_labels_test)
        prec, recall, f1, df_output = get_qald_metrics(test_pred, m_labels_test, ques_test, mode='test')
        df_output.to_csv(output_file_name)
        print("Test -- f1 is {} w/ threshold {}".format(f1, best_tuned_threshold))
        print("Test -- prec, recall, f1", prec, recall, f1)
        best_scores['precision'] = prec
        best_scores['recall'] = recall
        best_scores['f1'] = f1
        best_scores['threshold'] = best_tuned_threshold

    return test_pred, best_scores


def pick_model(model_name, alpha):
    if model_name == "purename":
        return PureNameLNN(alpha, -1, False)
    elif model_name == "context":
        return ContextLNN(alpha, -1, False)
    elif model_name == "complex":
        return ComplexRuleLNN(alpha, -1, False)
    else:
        print("WRONG name input")
        return None


def main():

    df_train_val = pd.read_csv(args.train_data)
    df_test = pd.read_csv(args.test_data)

    # train
    features_train_val = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_train_val.Features.values])
    X_train_val = torch.from_numpy(features_train_val).float()
    Y_train_val = torch.from_numpy(df_train_val.Label.values).float()
    mention_labels_train_val = df_train_val.Mention_label.values
    questions_train_val = df_train_val.Question.values
    x_train, x_val, y_train, y_val, m_labels_train, m_labels_val, ques_train, ques_val = \
        train_test_split(X_train_val, Y_train_val, mention_labels_train_val, questions_train_val,
                         test_size=0.2, train_size=0.8, random_state=100)

    # test
    features_test = np.array([np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_test.Features.values])
    x_test = torch.from_numpy(features_test).float()
    y_test = torch.from_numpy(df_test.Label.values).float()
    m_labels_test = df_test.Mention_label.values
    ques_test = df_test.Question.values

    # train model and evaluate
    model = pick_model(args.model_name, args.alpha)
    print(model(x_train, m_labels_train))
    print(x_train.shape, x_val.shape)
    # only pass ques_val for evaluation, not ques_train
    # best_tuned_threshold = train(model, x_train, y_train, m_labels_train, x_val, y_val, m_labels_val, ques_val, args.checkpoint_name, args.num_epoch)
    best_tuned_threshold = -1
    test_pred, best_scores = test(x_test, m_labels_test, ques_test, best_tuned_threshold, args.alpha, args.checkpoint_name, args.model_name, args.output_file_name)
    with open("output_w_spacy.txt", 'a') as f:
        f.write("model={}; p={}; r={}; f1={}; threshold={}\n".format(args.model_name,
                                                                    best_scores['precision'], best_scores['recall'],
                                                                    best_scores['f1'], best_scores['threshold']))
        print("model={}; p={}; r={}; f1={}; threshold={}\n".format(args.model_name,
                                                                    best_scores['precision'], best_scores['recall'],
                                                                    best_scores['f1'], best_scores['threshold']))


if __name__ == '__main__':
    main()