"""
scan_file
~~~~~~~~~

Can be used to scan a file and output results of ventmode classifications
"""
import argparse

import pandas as pd

from ventmode import datasets
from ventmode.main import merge_periods_with_low_time_thresh, perform_lookahead_confidence_window, run_dataset_with_classifier


def scan_file(filepath, cls_path, scaler_path, time_thresh):
    fileset = {'x': [('cli', filepath)]}
    vfinal = datasets.VFinalFeatureSet(fileset, 10, 100)

    df = vfinal.create_prediction_df()
    cls = pd.read_pickle(cls_path)
    scaler = pd.read_pickle(scaler_path)
    df = run_dataset_with_classifier(cls, scaler, df, "vfinal")
    map_ = {0: 'vc', 1: 'pc', 3: 'ps', 4: 'cpap', 6: 'pav'}
    df.predictions = perform_lookahead_confidence_window(df.predictions, df.patient, 30, .6)
    df.predictions = merge_periods_with_low_time_thresh(df.predictions, df.patient, df.abs_bs, pd.Timedelta(minutes=time_thresh))

    prev_mode = None
    bn_count = []
    for idx, row in df.iterrows():
        cur_mode = row.predictions
        if cur_mode != prev_mode and bn_count:
            print("mode: {}, breaths: {}-{}".format(map_[prev_mode], min(bn_count), max(bn_count)))
            bn_count = []
        bn_count.append(row.rel_bn)
        prev_mode = cur_mode
    else:
        print("mode: {}, breaths: {}-{}".format(map_[prev_mode], min(bn_count), max(bn_count)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vwd_file')
    parser.add_argument('saved_classifier_path')
    parser.add_argument('saved_scaler_path')
    parser.add_argument('-t', '--time-thresh', type=int, default=4, help='if a ventmode period is below this amount of time merge it with the closest neightboring mode')

    args = parser.parse_args()

    scan_file(args.vwd_file, args.saved_classifier_path, args.saved_scaler_path, args.time_thresh)


if __name__ == "__main__":
    main()
