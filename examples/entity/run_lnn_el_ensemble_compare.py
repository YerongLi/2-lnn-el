"""
python run_lnn_el_ensemble.py \
--dataset_name lcquad \
--experiment_name exp_lnn_lcquad_ensemble \
--model_name ensemble_with_type \
--learning_rate 0.0001 \
--use_type \
--topk 5

python run_lnn_el_ensemble.py \
--dataset_name lcquad \
--experiment_name exp_lnn_lcquad_ensemble \
--model_name ensemble_without_type \
--learning_rate 0.0001

python run_lnn_el_ensemble.py \
--dataset_name qald9 \
--experiment_name exp_lnn_qald9_ensemble \
--model_name ensemble_with_type \
--learning_rate 0.00001 \
--use_type \
--topk 5


python run_lnn_el_ensemble.py \
--dataset_name qald9 \
--experiment_name exp_lnn_qald9_ensemble \
--model_name ensemble_without_type \
--learning_rate 0.00001

python run_lnn_el_ensemble.py \
--dataset_name webqsp \
--experiment_name exp_lnn_webqsp_ensemble \
--model_name ensemble_with_type \
--learning_rate 0.00001 \
--use_type \
--topk 5

python run_lnn_el_ensemble.py \
--dataset_name webqsp \
--experiment_name exp_lnn_webqsp_ensemble \
--model_name ensemble_without_type \
--learning_rate 0.00001
"""

"""
python run_lnn_el_ensemble.py \
--dataset_name lcquad \
--experiment_name exp_lnn_blink_box_bert_lcquad_ensemble \
--model_name ensemble_with_type \
--learning_rate 0.0001 \
--margin 0.64 \
--num_epoch 50 \
--use_type \
--topk 5

python run_lnn_el_ensemble.py \
--dataset_name webqsp \
--experiment_name exp_lnn_blink_box_bert_webqsp_ensemble \
--model_name ensemble_with_type \
--margin 0.64 \
--num_epoch 50 \
--learning_rate 0.0001 \
--use_type \
--topk 5


python run_lnn_el_ensemble.py \
--dataset_name qald9 \
--experiment_name exp_lnn_blink_box_bert_qald_ensemble \
--model_name ensemble_with_type \
--margin 0.64 \
--learning_rate 0.00001 \
--num_epoch 50 \
--use_type \
--topk 5
========================================

python run_lnn_el_ensemble.py \
--dataset_name lcquad \
--experiment_name exp_lnn_blink_box_lcquad_ensemble \
--model_name ensemble_with_type \
--margin 0.64 \
--learning_rate 0.0001 \
--num_epoch 50 \
--use_type

python run_lnn_el_ensemble.py \
--dataset_name webqsp \
--experiment_name exp_lnn_blink_box_webqsp_ensemble \
--model_name ensemble_with_type \
--margin 0.64 \
--learning_rate 0.0001 \
--num_epoch 50 \
--use_type

python run_lnn_el_ensemble.py \
--dataset_name qald9 \
--experiment_name exp_lnn_blink_box_qald_ensemble \
--model_name ensemble_with_type \
--learning_rate 0.00001 \
--num_epoch 50 \
--use_type
"""
eps = 1e-7
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
# random.seed(103)

# device = torch.device("cpu")
try:
    # imports the torch_xla package
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except:
    print('TPU not available')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
parser.add_argument("--experiment_name", type=str, default="exp_lnn_lcquad", help="checkpoint path")
parser.add_argument("--dataset_name", type=str, default="lcquad", help="which model we choose")
parser.add_argument("--model_name", type=str, default="complex_pure_ctx", help="which model we choose")
parser.add_argument("--use_type", action="store_true", help="default is to use binary`, otherwise use stem")
parser.add_argument('--alpha', type=float, default=0.9, help='alpha for LNN')
parser.add_argument('--num_epoch', type=int, default=30, help='training epochs for LNN')
parser.add_argument('--margin', type=float, default=0.601, help='margin for MarginRankingLoss')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument("--topk", type=int, default=5)
parser.add_argument("--type", type=str, default=None)
# parser.add_argument("--use_fixed_threshold", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_refcount", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_blink", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--use_only_blink_candidates", action="store_true", help="default is to use binary`, otherwise use stem")
# parser.add_argument("--checkpoint_name", type=str, default="checkpoint/best_model.pt", help="checkpoint path")
# parser.add_argument("--log_file_name", type=str, default="log.txt", help="log_file_name")
parser.add_argument("-f")  # to avoid breaking jupyter noteOrganization
args = parser.parse_args()


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
    for x_0, y_0, qm_0, db_mask0, blink_mask0 in loader:
        x_ = x_0.to(device)
        y_ = y_0.to(device)
        qm_ = qm_0.to(device)
        db_mask = db_mask0.to(device)
        blink_mask = blink_mask0.to(device)
        # print('x_', x_)
        # print('y_', y_)
        # print('qm_', qm_)
        # print('db_mask', db_mask)
        # print('blink_mask', blink_mask)
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
            x_batch = torch.zeros(batch_length, batch_feature_len).to(device)
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
            yhat = model(xbatch_list[0], xbatch_list[1].to(device))
            extended_batch_length = len(yhat) - 1
            yhat_neg = yhat[:-1].to(device)
            yhat_pos = yhat[-1].repeat(extended_batch_length, 1).to(device)
            loss = loss_fn(yhat_pos, yhat_neg, torch.ones((len(yhat) - 1), 1).to(device))
            batch_loss_list.append(loss)
            total_loss += loss.item() * extended_batch_length
            count_batches += 1
        else:
            # get yhats for both BLINK and DB batches
            # print(len(xbatch_list[0]), len(xbatch_list[1]))
            # print((xbatch_list[0], xbatch_list[1]))
            yhat = model(xbatch_list[0], xbatch_list[1]).to(device)
            extended_batch_length = len(yhat) - 2
            yhat_neg = torch.zeros(extended_batch_length, 1).to(device)
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
    (x_train, y_train, df_train) = train_data
    (x_val, y_val, df_val) = val_data
    (x_test, y_test, df_test) = test_data
    qm_tensors_train, db_mask_train, blink_mask_train = convert_values_to_tensors(df_train)
    qm_tensors_test, db_mask_test, blink_mask_test = convert_values_to_tensors(df_test)

    # initialize the loss function and optimizer
    loss_fn = nn.MarginRankingLoss(margin=margin)  # MSELoss(), did not work neither
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # data loader
    dataset = TensorDataset(x_train, y_train, qm_tensors_train, db_mask_train, blink_mask_train)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(qm_tensors_train))), qm_tensors_train, False)
    loader = DataLoader(dataset, batch_sampler=question_sampler, shuffle=False)

    # start training
    print("=========TRAINING============")
    best_val_f1, best_val_loss, best_train_val_loss, best_model = 0, math.inf,  math.inf, None
    best_test_f1, best_test_loss = 0, math.inf
    for epoch in range(num_epochs):
        model.train()
        train_loss, batch_length = compute_loss(model, loader, loss_fn, optimizer)

        # show status after each epoch
        print("Epoch " + str(epoch) + ' / ' + str(num_epochs) + ": avg train loss -- " + str(train_loss))
        val_loss, val_f1, val_pred, _ = evaluate_ranking_loss(model, x_val, y_val, df_val, loss_fn, mode='val')
        test_loss, test_f1, test_pred, df_output = \
            evaluate_ranking_loss(model, x_test, y_test, df_test, loss_fn, mode='test')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        # if val_f1 > best_val_f1 + eps:
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and test_f1 >= best_test_f1):
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


def evaluate_ranking_loss(eval_model, x_, y_, df_, loss_fn, mode='val', hybrid = False):
    """
        evaluate a model on validation data
        Parameters:
            eval_model: Prediction model
            x_: feature vectors TODO remove x_
            y_: labels          TODO remove y_
            df_: Testing / Evaluation dataframe
    """

    qm_tensors_, db_mask_, blink_mask_ = convert_values_to_tensors(df_)
    dataset = TensorDataset(x_, y_, qm_tensors_, db_mask_, blink_mask_)
    question_sampler = QuestionSampler(torch.utils.data.SequentialSampler(range(len(qm_tensors_))),
                                       qm_tensors_, False)
    '''
        Batches are grouped by questions
    '''
    loader = DataLoader(dataset, batch_sampler=question_sampler, shuffle=False)

    eval_model.eval()
    with torch.no_grad():
        pred_ = []
        # cnt = 0
        for x_0, y_0, qm_0, db_mask0, blink_mask0 in loader:
            # print(x_0)
            x_ = x_0.to(device)
            y_ = y_0.to(device)
            qm_ = qm_0.to(device)
            db_mask = db_mask0.to(device)
            blink_mask = blink_mask0.to(device)
            # print('db_mask.shape. blink_mask.shape', db_mask.shape, blink_mask.shape)
            # print('db_mask', db_mask, blink_mask)
            xbatch_list = []
            xbatch_idxes = []
            for mask in [db_mask, blink_mask]:
                idxes = get_idxes_from_mask(mask)
                xbatch_idxes.append(idxes)
                print('mask', mask)
                print(idxes, xbatch_idxes)
                x_pick, y_pick, qm_pick = x_[idxes], y_[idxes], qm_[idxes]

                if len(x_pick) == 0:
                    xbatch_list.append(torch.tensor([]).to(device))
                    continue
                # elif len(x_pick) <= 1:
                #     xbatch_list.append(torch.tensor([]))
                #     continue
                x_batch = x_pick
                xbatch_list.append(x_batch)
            if (len(xbatch_list[0]) == 0) and (len(xbatch_list[1]) == 0):
                print(x_, y_, qm_, db_mask, blink_mask)
                print(xbatch_idxes)
            # # TODO Remove this line
            # xbatch_list = xbatch_list.to(device)
            # if not (xbatch_list[0].is_cuda and xbatch_list[1].is_cuda):
            #     print(xbatch_list[0], xbatch_list[1], 'xbatch_list')
            # print(eval_model, 'eval_model')
            yhat = eval_model(xbatch_list[0], xbatch_list[1])
            # get individual instance predicted prob
            yhat_in_order1 = torch.zeros(len(x_), 1).to(device)
            yhat_in_order2 = torch.zeros(len(x_), 1).to(device)
            yhat_in_order1[xbatch_idxes[0]] = yhat[:len(xbatch_idxes[0])]
            yhat_in_order2[xbatch_idxes[1]] = yhat[len(xbatch_idxes[0]):]
            yhat_in_order = torch.max(torch.cat((yhat_in_order1, yhat_in_order2), 1), 1)[0].reshape(-1,1)
            # print('yhat_in_order', yhat_in_order)
            pred_.append(yhat_in_order)
        pred_ = torch.cat(pred_, 0)
        prec, recall, f1, df_output = compute_qald_metrics(pred_, df_, gold_file_name=args.gold_file_name, topk=args.topk)
        # avg_loss, _ = compute_loss(eval_model, loader, loss_fn, optimizer=None)
        avg_loss = 0
        print("Current {} -- prec {}; recall {}; f1 {}; loss {}".format(mode, prec, recall, f1, avg_loss))

    return avg_loss, f1, pred_, df_output


def main():
    """Runs an experiment on pre-defined train/val/test split"""

    # set up output directory and file
    output_file_folder = "output/{}".format(args.experiment_name)
    Path(output_file_folder).mkdir(parents=True, exist_ok=True)
    args.output_file_name = "{}/{}.csv".format(output_file_folder, args.model_name)
    args.checkpoint_name = "{}/{}.pt".format(output_file_folder, args.model_name + "_best_model")
    equalization = True
    # TYPE = None
    RAWTYPE = args.type
    TYPE = ''.join([i for i in RAWTYPE if not i.isdigit()])

    # read lcquad merged data
    if args.dataset_name == "lcquad":
        # df_train = pd.read_csv("./data/lcquad/blink_bert_box//train_gold.csv")
        if equalization:
            k = 0.17
            if TYPE is not None:
                li = [
                    pd.read_csv(f"./data/lcquad/blink_bert_box//{TYPE}_train.csv"),
                    pd.read_csv(f"./data/lcquad/blink_bert_box//{TYPE}_valid.csv"),
                    # pd.read_csv("./data/lcquad/blink_bert_box//train_gold.csv"),
                    # pd.read_csv("./data/lcquad/blink_bert_box//valid_gold.csv"),
                ]
            else:
                li = [
                    pd.read_csv("./data/lcquad/blink_bert_box//train_gold.csv"),
                    pd.read_csv("./data/lcquad/blink_bert_box//valid_gold.csv"),
                ]              
            df_train = pd.concat(li, axis=0, ignore_index=True)

            df_train.drop_duplicates()
            qmset = set(df_train['QuestionMention'].unique())
            validqm = random.sample(qmset, int(len(qmset)* k))
            df_valid = df_train[df_train['QuestionMention'].isin(validqm)]
            df_train = df_train[~df_train['QuestionMention'].isin(validqm)]
            # df_valid = pd.read_csv("./data/lcquad/blink_bert_box//valid_gold.csv")
            # df_test = pd.read_csv("./data/lcquad/blink_bert_box//test_gold.csv")
        else:
            if TYPE is not None:

                df_train = pd.read_csv(f"./data/lcquad/blink_bert_box//{TYPE}_train.csv")
                df_valid = pd.read_csv(f"./data/lcquad/blink_bert_box//{TYPE}_valid.csv")
            else:
                df_train = pd.read_csv("./data/lcquad/blink_bert_box//train_gold.csv")
                df_valid = pd.read_csv("./data/lcquad/blink_bert_box//valid_gold.csv")
        
        # df_test = pd.read_csv("./data/lcquad/blink_bert_box//test_gold.csv")
        if TYPE is not None:
            df_test = pd.read_csv(f"./data/lcquad/blink_bert_box//{RAWTYPE}_test.csv")
            args.gold_file_name = f"./data/lcquad/{TYPE}_lcquad_gt_5000.csv"
        
        # args.gold_file_name = "./data/lcquad/lcquad_gt_5000.csv"
        else:
            df_test = pd.read_csv("./data/lcquad/blink_bert_box//test_gold.csv")
            args.gold_file_name = "./data/lcquad/full_lcquad_gt_5000.csv"
        # ### TEST
        # df_test = pd.read_csv("./data/lcquad/blink_bert_box//Food_test.csv")
        # args.gold_file_name = "./data/lcquad/Food_lcquad_gt_5000.csv"
        # ### TEST
    elif args.dataset_name == "qald9":
        df_train = pd.read_csv("./data/qald-9/blink_bert_box/train_gold.csv")
        df_valid = pd.read_csv("./data/qald-9/blink_bert_box/valid_gold.csv")
        df_test = pd.read_csv("./data/qald-9/blink_bert_box/test_gold.csv")
        args.gold_file_name = "./data/qald9/qald_data_gt.csv"
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
                model = EnsembleTypeRule(args.alpha, -1, False, printout=True)
    else:
        model = EnsembleRule(args.alpha, -1, False)
    # print(type(model))
    model = model.to(device)
    print(f'Training / Evaluating on GPU : {next(model.parameters()).is_cuda}')
    print("model: ", args.model_name, args.alpha)
    
    # args.checkpoint_name = 'output/exp_lnn_lcquad_ensemble/fullensemble_with_type_best_model.pt'
    if TYPE is not None:
        args.checkpoint_name = f'output/exp_lnn_lcquad_ensemble/{TYPE}ensemble_with_type_best_model.pt'
    else:
        args.checkpoint_name = 'output/exp_lnn_lcquad_ensemble/fullensemble_with_type_best_model.pt'
    # args.checkpoint_name = '/home/yerong/Downloads/fullensemble_with_type_best_model.pt'
    
    # args.checkpoint_name = '/home/yerong/Downloads/Personensemble_with_type_best_model.pt.8'




    print(args.checkpoint_name)
    # Load model
    # model.load_state_dict(torch.load(args.checkpoint_name))

    # training
    # train(model, train_data, valid_data, test_data, args.checkpoint_name, args.num_epoch, args.margin, args.learning_rate)

    # evaluate
    model.load_state_dict(torch.load(args.checkpoint_name))
    model.eval()
    (x_test, y_test, df_test) = test_data
    # (x_test, y_test, ) = train_data
    loss_fn = nn.MarginRankingLoss(margin=args.margin)
    avg_loss, f1, pred_, df_output = evaluate_ranking_loss(model, x_test, y_test, df_test, loss_fn, mode="test")
    df_output.to_csv("{}/prediction_{}".format(output_file_folder, os.path.basename(args.gold_file_name)))


if __name__ == '__main__':
    main()