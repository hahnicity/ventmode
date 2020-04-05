"""
add_dtw_scores_to_dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

Pick a dataset and add dtw scores to them
"""
import argparse

import numpy as np
import pandas as pd

from ventmode.raw_utils import extract_raw
from ventmode.dtw_lib import dtw_file_analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='path to input dataset')
    parser.add_argument('output', help='filepath to output dataset')
    parser.add_argument('-n', '--n-breaths', type=int, default=16)
    parser.add_argument('-w', '--rolling-av-window-len', type=int, default=10)
    args = parser.parse_args()

    df = pd.read_pickle(args.dataset)
    df['dtw'] = np.nan
    df.index = range(len(df))
    for filename in df.x_filename.unique():
        gen = extract_raw(open(filename), False)
        scores, rel_bns = dtw_file_analyze(gen, args.n_breaths, args.rolling_av_window_len)
        file_df = df[df.x_filename == filename]
        can_have_dtw_score = file_df[file_df.rel_bn.isin(rel_bns)]
        compr_rel_bns = can_have_dtw_score.rel_bn.values
        final_scores = []
        for idx, score in enumerate(scores):
            if rel_bns[idx] in compr_rel_bns:
                final_scores.append(score)
        df.loc[can_have_dtw_score.index, 'dtw'] = final_scores
    df.to_pickle(args.output)


if __name__ == "__main__":
    main()
