"""
run_sequential_simple_split
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because simple split yields varying model results based on the split just run
sequential models and take the average.
"""
import argparse

import pandas as pd
from prettytable import PrettyTable

from main import Run, build_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--from-pickle', required=True)
    parser.add_argument('-nr', '--num-runs', type=int, default=100)
    parser.add_argument('-sr', '--split-ratio', type=float, default=.2)
    parser.add_argument('--algo', choices=['svm', 'rf', 'log_reg', 'mlp'], default='rf')
    main_args = parser.parse_args()

    model_args = build_parser().parse_args([])
    model_args.from_pickle = main_args.from_pickle
    model_args.split_type = 'simple'
    model_args.test_split_ratio = main_args.split_ratio
    model_args.no_print_results = True
    model_args.algo = main_args.algo
    model_args.time_thresh_cutoff = 5
    model_args.conf_window_frac = .6
    model_args.with_lookahead_conf = 30

    df = pd.read_pickle(main_args.from_pickle)
    df.index = range(len(df))
    aggregated_results = []

    cols = ['cls', 'f1', 'sen', 'spec', 'prec']
    for i in range(main_args.num_runs):
        print("perform run {}".format(i+1))
        model = Run(model_args)
        model.run(df)
        for _, row in model.final_results.iterrows():
            aggregated_results.append(row[cols].tolist())

    aggregated_results = pd.DataFrame(aggregated_results, columns=cols)
    map = {0.0: 'vc', 1.0: 'pc', 3.0: 'ps', 4.0: 'cpap', 6.0: 'pav'}
    table = PrettyTable()
    table.field_names = cols
    for label in sorted(aggregated_results.cls.unique()):
        label_results = aggregated_results[aggregated_results.cls == label]
        table.add_row([map[label]]+label_results[cols[1:]].mean(axis=0).tolist())
    aggregated_results.to_pickle('sequential_run_results.pkl')
    print(table)


if __name__ == "__main__":
    main()
