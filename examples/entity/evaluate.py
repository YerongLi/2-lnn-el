import sys

sys.path.append('../../../')
import glob, os
import pandas as pd
from el_evaluation import *
import numpy as np
from pathlib import Path
import argparse


def calculate_qald_metric(df_output, use_topk=False, analysis_files_path=None):
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

    if use_topk:
        metrics = compute_metrics_top_k(benchmark=benchmark, predictions=predictions,
                                        analysisfile=analysis_files_path, limit=414,
                                        is_qald_gt=is_qald_gt, eval='full')
    #         print(metrics)
    else:
        metrics = compute_metrics(benchmark=benchmark, predictions=predictions,
                                  limit=410, is_qald_gt=is_qald_gt, eval='full')

    scores = metrics['macro']['named']
    prec, recall, f1 = scores['precision'], scores['recall'], scores['f1']
    return prec, recall, f1, df_output


parser = argparse.ArgumentParser(description="main training script for training lnn entity linking models")
parser.add_argument("--experiment_prediction_foldername", type=str, default="complex_pure_ctx_type")
parser.add_argument("--missing_file_path", type=str, default="data/type_missing.csv")
parser.add_argument("-f")  # to avoid breaking jupyter notebook
args = parser.parse_args()


def main():
    experiment_name = args.experiment_prediction_foldername
    output_dir = "output/{}/".format(experiment_name)
    files = sorted(glob.glob(os.path.join(output_dir, "*.csv")), key=lambda x: int(x.split('_')[-1][:-4]))
    print(files)

    exp_folder_adding_missing = "{}_with_missing_sentences".format(experiment_name)
    Path(os.path.join("output", exp_folder_adding_missing)).mkdir(parents=True, exist_ok=True)

    # load missing csv
    df_missing = pd.read_csv(args.missing_file_path, header=None)
    df_missing.columns = ['Unnamed:0', 'Question', 'Entities', 'Classes']
    df_missing = df_missing[['Question', 'Entities', 'Classes']]

    missing_assignments = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10]]

    topk_performance_list = []
    top1_performance_list = []

    for missingIdxes, filename in zip(missing_assignments, files):
        df_output = pd.read_csv(filename)
        df_output = df_output[['Question', 'Entities', 'Classes']]
        print(">> before adding missing, df_output.shape:", df_output.shape)

        df_missing_assigned = df_missing.iloc[missingIdxes]
        df_output = pd.concat([df_output, df_missing_assigned], ignore_index=True)
        df_output.to_csv("./output/{}/{}".format(exp_folder_adding_missing, filename.split('/')[-1]))
        print(">> after adding missing, df_output.shape:", df_output.shape)

        prec, recall, f1, df_output = calculate_qald_metric(df_output, use_topk=True,
                                                            analysis_files_path="analysis.csv")
        topk_performance_list.append([prec, recall, f1])
        print(">> top K: ", prec, recall, f1)

        prec, recall, f1, df_output = calculate_qald_metric(df_output, use_topk=False)
        top1_performance_list.append([prec, recall, f1])
        print(">>top 1: ", prec, recall, f1)

    topk_performance_list = np.array(topk_performance_list)
    print(">> top k avg: ", topk_performance_list.mean(0))

    top1_performance_list = np.array(top1_performance_list)
    print(">> top 1 avg: ", top1_performance_list.mean(0))


if __name__ == '__main__':
    main()
