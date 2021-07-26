import sys, copy, random, math, argparse, os
sys.path.append('../../../')
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
from utils import QuestionSampler, read_data_file, compute_qald_metrics, convert_values_to_tensors
import torch
from torch import nn, optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models.LNN_EL_merge import ComplexRuleDbpediaBLINK, EnsembleRule, EnsembleTypeRule, EnsembleTypeRuleWebSQP
from models.LNN_EL_merge import EnsembleBLINKBoxRule, EnsembleBLINKBoxBertRule
from models.LNN_EL_merge import EnsembleBLINKBoxRuleWebQSP, EnsembleBLINKBoxBertRuleWebQSP


import pandas as pd
random.seed(103)

device = torch.device("cpu")

def evaluate_ranking_loss(eval_model, x_, y_, df_, loss_fn, mode='val'):
    """evaluate a model on validation data"""

    qm_tensors_, db_mask_, blink_mask_ = convert_values_to_tensors(df_)
    dataset = TensorDataset(x_, y_, qm_tensors_, db_mask_, blink_mask_)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(qm_tensors_))),
                                       qm_tensors_, False)
    loader = DataLoader(dataset, batch_sampler=question_sampler, shuffle=False)

    eval_model.eval()
    with torch.no_grad():
        pred_ = []
        for x_, y_, qm_, db_mask, blink_mask in loader:
            xbatch_list = []
            xbatch_idxes = []
            for mask in [db_mask, blink_mask]:
                idxes = get_idxes_from_mask(mask)
                xbatch_idxes.append(idxes)
                x_pick, y_pick, qm_pick = x_[idxes], y_[idxes], qm_[idxes]

                if len(x_pick) == 0:
                    xbatch_list.append(torch.tensor([]))
                    continue
                # elif len(x_pick) <= 1:
                #     xbatch_list.append(torch.tensor([]))
                #     continue
                x_batch = x_pick
                xbatch_list.append(x_batch)
            if (len(xbatch_list[0]) == 0) and (len(xbatch_list[1]) == 0):
                print(x_, y_, qm_, db_mask, blink_mask)
                print(xbatch_idxes)
            yhat = eval_model(xbatch_list[0], xbatch_list[1])
            # get individual instance predicted prob
            yhat_in_order1 = torch.zeros(len(x_), 1)
            yhat_in_order2 = torch.zeros(len(x_), 1)
            yhat_in_order1[xbatch_idxes[0]] = yhat[:len(xbatch_idxes[0])]
            yhat_in_order2[xbatch_idxes[1]] = yhat[len(xbatch_idxes[0]):]
            yhat_in_order = torch.max(torch.cat((yhat_in_order1, yhat_in_order2), 1), 1)[0].reshape(-1,1)
            # print(yhat_in_order)
            pred_.append(yhat_in_order)
        pred_ = torch.cat(pred_, 0)
        prec, recall, f1, df_output = compute_qald_metrics(pred_, df_, gold_file_name=args.gold_file_name, topk=args.topk)
        avg_loss, _ = compute_loss(eval_model, loader, loss_fn, optimizer=None)
        print("Current {} -- prec {}; recall {}; f1 {}, loss {}".format(mode, prec, recall, f1, avg_loss))

    return avg_loss, f1, pred_, df_output


def main():
    """Runs an experiment on pre-defined train/val/test split"""

    # set up output directory and file
    output_file_folder = "output/{}".format(args.experiment_name)
    Path(output_file_folder).mkdir(parents=True, exist_ok=True)
    args.output_file_name = "{}/{}.csv".format(output_file_folder, args.model_name)
    args.checkpoint_name = "{}/{}.pt".format(output_file_folder, args.model_name + "_best_model")

    # read lcquad merged data
    if args.dataset_name == "lcquad":
        df_train = pd.read_csv("./data/lcquad/blink_bert_box//train_gold.csv")
        df_valid = pd.read_csv("./data/lcquad/blink_bert_box//valid_gold.csv")
        df_test = pd.read_csv("./data/lcquad/blink_bert_box//test_gold.csv")
        args.gold_file_name = "lcquad/lcquad_gt_5000.csv"
    elif args.dataset_name == "qald9":
        df_train = pd.read_csv("./data/qald-9/blink_bert_box/train_gold.csv")
        df_valid = pd.read_csv("./data/qald-9/blink_bert_box/valid_gold.csv")
        df_test = pd.read_csv("./data/qald-9/blink_bert_box/test_gold.csv")
        args.gold_file_name = "qald/qald_data_gt.csv"
    elif args.dataset_name == "webqsp":
        df_train = pd.read_csv("./data/webqsp/blink_bert_box/train_gold.csv")
        df_valid = pd.read_csv("./data/webqsp/blink_bert_box/valid_gold.csv")
        df_test = pd.read_csv("./data/webqsp/blink_bert_box/test_gold.csv")
        args.gold_file_name = "webqsp/webqsp_data_gt.csv"

    train_data = read_data_file(df_train, device, "train")
    valid_data = read_data_file(df_valid, device, "valid")
    test_data = read_data_file(df_test, device, "test")

    # train model and evaluate
    if args.use_type:
        if args.dataset_name == "webqsp":
            if 'blink_box_bert' in args.experiment_name:
                model = EnsembleBLINKBoxBertRuleWebQSP(args.alpha, -1, False)
            elif "blink_box" in args.experiment_name:
                model = EnsembleBLINKBoxRuleWebQSP(args.alpha, -1, False)
            else:
                model = EnsembleTypeRuleWebSQP(args.alpha, -1, False)
        else:
            if 'blink_box_bert' in args.experiment_name:
                model = EnsembleBLINKBoxBertRule(args.alpha, -1, False)
            elif "blink_box" in args.experiment_name:
                model = EnsembleBLINKBoxRule(args.alpha, -1, False)
            else:
                model = EnsembleTypeRule(args.alpha, -1, False)
    else:
        model = EnsembleRule(args.alpha, -1, False)
    model = model.to(device)
    print("model: ", args.model_name, args.alpha)

    # training
    train(model, train_data, valid_data, test_data, args.checkpoint_name, args.num_epoch, args.margin, args.learning_rate)

    # evaluate
    model.load_state_dict(torch.load(args.checkpoint_name))
    model.eval()
    (x_test, y_test, df_test) = test_data
    loss_fn = nn.MarginRankingLoss(margin=args.margin)
    avg_loss, f1, pred_, df_output = evaluate_ranking_loss(model, x_test, y_test, df_test, loss_fn, mode="test")
    df_output.to_csv("{}/prediction_{}".format(output_file_folder, os.path.basename(args.gold_file_name)))


if __name__ == '__main__':
    main()