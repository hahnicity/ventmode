"""
dataset_ablation_collate
~~~~~~~~~~~~~~~~~~~~~~~~

collate all results from dataset ablation
"""
import argparse
import multiprocessing
import random

import numpy as np
import pandas as pd

from ventmode.main import build_parser, Run

dtw_threshs = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500, 800, 1000, 1500, 2000]
dtw_n_lookback = [8, 16]
random_threshs = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .96, .97, .98, .99, .991, .992, .993, .994, .995, .996, .997, .998, .999]
patient_ablation = [.1, .2, .3, .4, .5, .6, .7, .8, .9]


def run_ablation(main_args, file_to_use, cls_specific_reporting, results_collated, dataset_pickle_file):
    runner = Run(main_args)
    reference_df = pd.read_pickle(dataset_pickle_file)
    try:
        df = pd.read_pickle(file_to_use)
    except Exception as err:
        print(err)
        return results_collated
    df.index = range(len(df))
    try:
        runner.run(df)
    except Exception as err:
        runner.final_results.loc[0] = ([np.nan] * 14)
    for _, row in runner.final_results.iterrows():
        cls_orig = reference_df[(reference_df.y == row.cls) & (reference_df.set_type == 'train')]
        if len(cls_orig) == 0:
            ablation_frac = np.nan
        else:
            ablation_frac = 1 - (float(row.train_len) / len(cls_orig))
        i = len(results_collated)
        results_collated.loc[i] = cls_specific_reporting + row.tolist() + [len(cls_orig), ablation_frac]
    return results_collated


def dtw_func_star(args):
    return dtw_multi(*args)


def dtw_multi(thresh, main_args, file_fmt, dataset_pickle_file):
    i = 0
    collated = pd.DataFrame([], columns=['dtw_thresh', 'dtw_n', 'f1', 'acc', 'sen', 'spec', 'prec', 'train_len', 'test_len', 'n_train_pts', 'n_test_pts', 'tns', 'tps', 'fns', 'fps', 'orig_train_len', 'ablation'])
    for n in dtw_n_lookback:
        file_to_use = file_fmt.format(thresh, n)
        collated = run_ablation(main_args, file_to_use, [thresh, n], collated, dataset_pickle_file)
    return collated


def random_func_star(args):
    return random_multi(*args)


def random_multi(rand_thresh, main_args, file_fmt, dataset_pickle_file):
    collated = pd.DataFrame([], columns=['rand_thresh', 'cls', 'f1', 'acc', 'sen', 'spec', 'prec', 'train_len', 'test_len', 'n_train_pts', 'n_test_pts', 'tns', 'tps', 'fns', 'fps', 'orig_train_len', 'ablation'])
    float_fmt = str(rand_thresh)[1:]
    file_to_use = file_fmt.format(float_fmt)
    return run_ablation(main_args, file_to_use, [rand_thresh], collated, dataset_pickle_file)


def perform_collate(main_args, script_args, file_fmt, output_fmt, collate_type, dataset_pickle_file):
    pool = multiprocessing.Pool(script_args.threads)
    if collate_type == 'dtw':
        func = dtw_func_star
        input_gen = [(thresh, main_args, file_fmt, dataset_pickle_file) for thresh in dtw_threshs]
    elif collate_type == 'random':
        func = random_func_star
        input_gen = [(thresh, main_args, file_fmt, dataset_pickle_file) for thresh in random_threshs]
    results = pool.map(func, input_gen)
    pool.close()
    pool.join()
    collated = pd.concat(results)
    collated.index = range(len(collated))
    collated.to_pickle(output_fmt.format(main_args.algo))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='filepath to pickled dataset')
    parser.add_argument('ablation_type', choices=['random', 'dtw', 'patient', 'random_optimal', 'dtw_optimal'])
    parser.add_argument('--algo', choices=['svm', 'rf', 'log_reg', 'mlp'], default='rf')
    parser.add_argument('-t', '--threads', type=int, default=10)
    args = parser.parse_args()

    main_args = build_parser().parse_args([])
    main_args.no_print_results = True
    main_args.split_type = 'validation'
    main_args.lookahead_conf_frac = .6
    main_args.with_lookahead_conf = 50
    main_args.algo = args.algo

    if args.ablation_type == 'dtw':
        perform_collate(main_args, args, 'dtw-thresh{}-n{}.pkl', 'dtw_ablation_results_{}.pkl', 'dtw', args.dataset)
    elif args.ablation_type == 'dtw_optimal':
        perform_collate(main_args, args, 'dtw-thresh{}-n{}-with-optimal-reduction.pkl', 'dtw_ablation_results_with_optimal_reduction_{}.pkl', 'dtw', args.dataset)
    elif args.ablation_type == 'random':
        perform_collate(main_args, args, 'rand-thresh{}.pkl', 'random_ablation_results_{}.pkl', 'random', args.dataset)
    elif args.ablation_type == 'patient':
        perform_collate(main_args, args, 'rand-patient-thresh{}.pkl', 'random_patient_ablation_results_{}.pkl', 'random', args.dataset)
    elif args.ablation_type == 'random_optimal':
        perform_collate(main_args, args, 'rand-thresh{}-with-optimal-reduction.pkl', 'random_ablation_results_with_optimal_reduction_{}.pkl', 'random', args.dataset)


if __name__ == "__main__":
    main()
