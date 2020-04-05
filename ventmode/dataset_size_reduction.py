"""
dataset_size_reduction
~~~~~~~~~~~~~~~~~~~~~~

Reduce size of the dataset in a natural fashion and see how well we do
"""
import argparse
import random

import numpy as np
import pandas as pd

from ventmode.main import build_parser, Run

n_breaths = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400]


def run_sensitivity_analysis(df, main_args):
    # sanity check
    df.index = range(len(df))

    train_set = df[df.set_type == 'train']
    i = 0
    collated = pd.DataFrame([], columns=['n', 'ablation', 'cls', 'f1', 'acc', 'sen', 'spec', 'prec', 'train_len', 'test_len'])
    for cls in [0, 1, 3, 4, 6]:
        for n in n_breaths:
            n_df = df.copy()
            for filename in train_set.x_filename.unique():
                file_df = df[df.x_filename == filename]
                mode_in_file = file_df[file_df.y == cls]
                not_first_n = mode_in_file.iloc[n:].index
                n_df = n_df.loc[n_df.index.difference(not_first_n)]
            runner = Run(main_args)
            try:
                runner.run(n_df)
            except:
                runner.final_results.loc[0] = ([np.nan] * 14)
            final_results = runner.final_results
            row = final_results[final_results.cls == cls].iloc[0]
            ablation = 1 - (row.train_len / float(len(train_set[train_set.y == row.cls])))
            collated.loc[i] = [n, ablation] + [row.cls, row.f1, row.acc, row.sen, row.spec, row.prec, row.train_len, row.test_len]
            i += 1
    collated.to_pickle('contiguous-ablation-results.pkl')


def run_optimal_ablation_analysis(df, main_args):
    # sanity check
    df.index = range(len(df))

    train_set = df[df.set_type == 'train'].copy()
    collated = pd.DataFrame([], columns=['ablation', 'cls', 'f1', 'acc', 'sen', 'spec', 'prec', 'train_len', 'test_len'])
    for filename in train_set.x_filename.unique():
        for n, y in [(450, 0), (120, 1), (1200, 3), (160, 4), (80, 6)]:
        # old results
        #for n, y in [(450, 0), (400, 1), (400, 3), (70, 4), (300, 6)]:
            file_df = df[df.x_filename == filename]
            mode_in_file = file_df[file_df.y == y]
            not_first_n = mode_in_file.iloc[n:].index
            df = df.loc[df.index.difference(not_first_n)]
    runner = Run(main_args)
    try:
        runner.run(df)
    except:
        runner.final_results.loc[0] = ([np.nan] * 14)
    for i, row in runner.final_results.iterrows():
        ablation = 1 - (row.train_len / float(len(train_set[train_set.y == row.cls])))
        collated.loc[i] = [ablation] + [row.cls, row.f1, row.acc, row.sen, row.spec, row.prec, row.train_len, row.test_len]
    print(collated)
    collated.to_pickle('optimal-ablation-results.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--from-pickle', required=True)
    parser.add_argument('--algo', choices=['svm', 'rf', 'log_reg', 'mlp'], default='rf')
    parser.add_argument('-t', '--analysis-type', choices=['sensitivity', 'optimal'], required=True)
    args = parser.parse_args()
    df = pd.read_pickle(args.from_pickle)

    main_args = build_parser().parse_args([])
    main_args.no_print_results = True
    main_args.split_type = 'validation'
    main_args.algo = args.algo
    main_args.lookahead_conf_frac = .6
    main_args.with_lookahead_conf = 50
    if args.analysis_type == 'sensitivity':
        run_sensitivity_analysis(df, main_args)
    elif args.analysis_type == 'optimal':
        run_optimal_ablation_analysis(df, main_args)


if __name__ == "__main__":
    main()
