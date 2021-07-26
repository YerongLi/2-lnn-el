"""
python run_lnn_el_ablation.py \
--dataset_name lcquad \
--experiment_name exp_lnn_lcquad_pure \
--model_name pure \
--learning_rate 0.0001

"""
import sys, copy, random, math, argparse
sys.path.append('../../../')
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
from utils import QuestionSampler, compute_qald_metrics, convert_values_to_tensors
import torch
from torch import nn, optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from models.LNN_EL_merge import ComplexRuleDbpediaBLINK, EnsembleRule, EnsembleTypeRule
from models.LNN_EL_gold_db import *
import pandas as pd
random.seed(103)

device = torch.device("cpu")

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
parser.add_argument("--experiment_name", type=str, default="exp_lnn_lcquad", help="checkpoint path")
parser.add_argument("--dataset_name", type=str, default="lcquad", help="which model we choose")
parser.add_argument("--model_name", type=str, default="complex_pure_ctx", help="which model we choose")
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=100, help='training epochs for LNN')
parser.add_argument('--margin', type=float, default=0.601, help='margin for MarginRankingLoss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
# parser.add_argument("--use_fixed_threshold", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_refcount", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_blink", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_only_blink_candidates", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
# parser.add_argument("--log_file_name", type=str, default="log.txt", help="log_file_name")
parser.add_argument("-f")  # to avoid breaking jupyter notebook
args = parser.parse_args()

def read_data_file(df_, device, split_name="train"):
    """read the lcquad data from DBpedia(not combined version)"""
    features_train = np.array(
        [np.fromstring(s[1:-1], dtype=np.float, sep=', ') for s in df_.Features.values])
    x_ = torch.from_numpy(features_train).float().to(device)
    y_ = torch.from_numpy(df_.Label.values).float().reshape(-1, 1).to(device)
    m_labels_ = df_.Mention_label.values
    ques_ = df_.QuestionMention.values
    print("loading {}".format(split_name))
    return x_, y_, m_labels_, ques_

def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, 'param -- data', param.data, 'grad -- ', param.grad)


def get_idxes_from_mask(mask):
    """mask is (batch_size, 1)"""
    if len(mask) > 1:
        return torch.nonzero(mask.squeeze(), as_tuple=False).reshape(1, -1)[0]
    elif len(mask) == 1:
        return torch.tensor([0], dtype=torch.int64) if mask.sum() == 1 else torch.tensor([], dtype=torch.int64)
    return torch.tensor([], dtype=torch.int64)


# def test(x_test, m_labels_test, ques_test, alpha, checkpoint_name, model_name, output_file_name):
#     """make predictions on test set"""
#     bestModel = ComplexRuleDbpediaBLINK(model_name, alpha)
#     bestModel.load_state_dict(torch.load(checkpoint_name))
#     bestModel.eval()
#     best_scores = {}
#
#     with torch.no_grad():
#         test_pred = bestModel(x_test, m_labels_test)
#         prec, recall, f1, df_output = compute_qald_metrics(test_pred, m_labels_test, ques_test, mode='test')
#         df_output.to_csv(output_file_name)
#         print("Test -- f1 is {} ".format(f1))
#         print("Test -- prec, recall, f1", prec, recall, f1)
#         best_scores['precision'] = prec
#         best_scores['recall'] = recall
#         best_scores['f1'] = f1
#
#     return test_pred, best_scores


def compute_loss(model, loader, loss_fn, optimizer=None):
    """compute a single loss update for a full pass
    if optimizer is not None, then we assume this is the training mode
    """

    total_loss = 0.0
    count_batches = 0
    for x_, y_, qm_, db_mask, blink_mask in loader:
        batch_loss_list = []
        xbatch_list = []
        for mask in [db_mask, blink_mask]:
            idxes = get_idxes_from_mask(mask)
            x_pick, y_pick, qm_pick = x_[idxes], y_[idxes], qm_[idxes]
            y_pos_idxes = torch.nonzero(y_pick.squeeze(), as_tuple=False).reshape(1, -1)[0]
            y_neg_idxes = torch.nonzero(~y_pick.squeeze().bool(), as_tuple=False).reshape(1, -1)[0]

            if (len(y_pos_idxes) == 0) or (len(y_neg_idxes) == 0):
                xbatch_list.append(torch.tensor([]))
                continue
            elif len(x_pick) <= 1:
                xbatch_list.append(torch.tensor([]))
                continue
            elif len(y_pos_idxes) == 1:
                y_pos_idx = y_pos_idxes[0]
            else:  # len(y_pos_idxes) > 1:
                # TODO: I am just always using the first positive example for now
                # rand_idx = random.choice(list(range(len(y_pos_idxes))))
                # print(y_pos_idxes)
                rand_idx = 0
                y_pos_idx = y_pos_idxes[rand_idx]

            batch_length = 1 + len(y_neg_idxes)
            batch_feature_len = x_.shape[1]
            x_batch = torch.zeros(batch_length, batch_feature_len)
            x_batch[:-1:, :] = x_pick[y_neg_idxes]
            x_batch[-1, :] = x_pick[y_pos_idx]  # put positive to the end
            xbatch_list.append(x_batch)
            # print(y_pos_idx, len(y_neg_idxes))
            # print("batch", x_batch.shape)

        if (len(xbatch_list[0]) == 0) and (len(xbatch_list[1]) == 0):
            # skip if both batches are []
            # print("hitting cases without any examples [SHOULD BE WRONG]")
            continue
        elif (len(xbatch_list[0]) == 0) or (len(xbatch_list[1]) == 0):
            # continue # TODO: testing whether improvements made if we only use cases where there are sources from both
            yhat = model(xbatch_list[0], xbatch_list[1])
            extended_batch_length = len(yhat) - 1
            yhat_neg = yhat[:-1]
            yhat_pos = yhat[-1].repeat(extended_batch_length, 1)
            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            batch_loss_list.append(loss)
            total_loss += loss.item() * extended_batch_length
            count_batches += 1
        else:
            # get yhats for both BLINK and DB batches
            # print(len(xbatch_list[0]), len(xbatch_list[1]))
            # print((xbatch_list[0], xbatch_list[1]))
            yhat = model(xbatch_list[0], xbatch_list[1])
            extended_batch_length = len(yhat) - 2
            yhat_neg = torch.zeros(extended_batch_length, 1)
            yhat_neg[:len(xbatch_list[0])-1] = yhat[:len(xbatch_list[0])-1]
            yhat_neg[len(xbatch_list[0])-1:] = yhat[len(xbatch_list[0]):-1]
            for idx in [len(xbatch_list[0]), -1]:
                yhat_pos = yhat[idx].repeat(extended_batch_length, 1)
                loss = loss_fn(yhat_pos, yhat_neg, torch.ones(extended_batch_length, 1).to(device))
                batch_loss_list.append(loss)
                total_loss += loss.item() * extended_batch_length
                count_batches += 1

        # update every question-mention
        if batch_loss_list and optimizer:
            (sum(batch_loss_list)/len(batch_loss_list)).backward()
            optimizer.step()

    avg_loss = total_loss / count_batches

    return avg_loss, batch_length


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

    # data loader
    dataset = TensorDataset(x_train, y_train)
    question_sampler = \
        QuestionSampler(torch.utils.data.SequentialSampler(range(len(ques_train))), ques_train, False)
    loader = DataLoader(dataset, batch_sampler=question_sampler, shuffle=False)

    # start training
    print("=========TRAINING============")
    best_val_f1, best_val_loss, best_train_val_loss, best_model = 0, math.inf,  math.inf, None
    best_test_f1, best_test_loss = 0, math.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss, batch_length = compute_loss(model, loader, loss_fn, optimizer)

        # show status after each epoch
        print("Epoch " + str(epoch) + ": avg train loss -- " + str(train_loss))
        val_loss, val_f1, val_pred, _ = evaluate_ranking_loss(model, x_val, y_val, df_val, loss_fn, mode='val')
        test_loss, test_f1, test_pred, df_output = \
            evaluate_ranking_loss(model, x_test, y_test, df_test, loss_fn, mode='test')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        if val_f1 >= best_val_f1:
            best_val_f1, best_val_loss= val_f1, val_loss
            best_test_f1, best_test_loss = test_f1, test_loss
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), checkpoint_name)
            df_output.to_csv(args.output_file_name)
        print("Val -- loss {}; F1 {}".format(val_loss, val_f1))
        print("Best Val --  best loss {}; F1 {}".format(best_val_loss, best_val_f1))
        print("Test --  best loss {}; F1 {}".format(test_loss, test_f1))
        print("Best Test --  best loss {}; F1 {}".format(best_test_loss, best_test_f1))

    # finished logging
    writer.close()

    return


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
        prec, recall, f1, df_output = compute_qald_metrics(pred_, df_, gold_file_name=args.gold_file_name)
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
        df_train = pd.read_csv("./data/lcquad/gold_db/train_gold.csv")
        df_valid = pd.read_csv("./data/lcquad/gold_db/valid_gold.csv")
        df_test = pd.read_csv("./data/lcquad/gold_db/lcquad_test_sorted.csv")
        args.gold_file_name = "lcquad/lcquad_gt_5000.csv"
    # elif args.dataset_name == "qald9":
    #     df_train = pd.read_csv("./data/qald-9/train_gold.csv")
    #     df_valid = pd.read_csv("./data/qald-9/valid_gold.csv")
    #     df_test = pd.read_csv("./data/qald-9/test_gold.csv")
    #     args.gold_file_name = "qald/qald_data_gt.csv"
    # elif args.dataset_name == "webqsp":
    #     df_train = pd.read_csv("./data/webqsp/train_gold.csv")
    #     df_valid = pd.read_csv("./data/webqsp/valid_gold.csv")
    #     df_test = pd.read_csv("./data/webqsp/test_gold.csv")
    #     args.gold_file_name = "webqsp/webqsp_data_gt.csv"

    train_data = read_data_file(df_train, device, "train")
    valid_data = read_data_file(df_valid, device, "valid")
    test_data = read_data_file(df_test, device, "test")

    # train model and evaluate
    if args.model_name == "pure":
        model = PureNameLNN(args.alpha, -1, False)
    elif args.model_name == "ctx":
        model = None
    elif args.model_name == 'type':
        model = None
    elif args.model_name == "pure_ctx":
        model = None
    elif args.model_name == "pure_type":
        model = None
    elif args.model_name == "ctx_type":
        model = None

    model = model.to(device)
    print("model: ", args.model_name, args.alpha)

    # training
    train(model, train_data, valid_data, test_data, args.checkpoint_name, args.num_epoch, args.margin, args.learning_rate)


if __name__ == '__main__':
    main()