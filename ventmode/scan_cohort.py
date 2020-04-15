import argparse
from itertools import groupby
import multiprocessing
import os
import pickle
from warnings import warn

import pandas as pd

from ventmode import datasets
from ventmode.main import InvalidVMPatientError, merge_periods_with_low_time_thresh, perform_lookahead_confidence_window, run_dataset_with_classifier

PATIENT_PATTERN = r'(\d{4}RPI\d{10})'


def analyze_patient(args, patient_id, patient_fs):
    # results should be structured like:
    # [[patientXXX, time start, time end, mode],
    #  [patientYYY, time start, time end, mode],
    #  ...
    #  ]
    results = []
    pickle_path = os.path.join(args.patient_results_path, patient_id + '.pkl')
    if os.path.exists(pickle_path) and args.skip_if_analyzed:
        return
    patient_fs = {'x': list(patient_fs)}
    vfinal = datasets.VFinalFeatureSet(patient_fs, 10, 100, PATIENT_PATTERN)
    df = vfinal.create_prediction_df()
    cls = pickle.load(open(args.saved_classifier_path))
    scaler = pickle.load(open(args.saved_scaler_path))
    df = run_dataset_with_classifier(cls, scaler, df, "vfinal")
    df['predictions'] = perform_lookahead_confidence_window(df.predictions, df.patient, 30, .6)
    try:
        df['predictions'] = merge_periods_with_low_time_thresh(df.predictions, df.patient, df.abs_bs, pd.Timedelta(minutes=args.time_thresh))
    except InvalidVMPatientError:
        return

    cur_mode = df.iloc[0].predictions
    start_time = df.iloc[0].abs_bs
    prev_mode = cur_mode
    for idx, row in df.iloc[1:].iterrows():
        cur_mode = row.predictions
        if cur_mode != prev_mode:
            results.append([patient_id, start_time, df.loc[last_idx].abs_bs, prev_mode])
            start_time = row.abs_bs
        last_idx = idx
        prev_mode = cur_mode
    else:
        results.append([patient_id, start_time, df.loc[idx].abs_bs, cur_mode])

    results = pd.DataFrame(results, columns=['patient', 'start_time', 'end_time', 'ventmode'])
    results.loc[results['ventmode'] == 0, 'ventmode'] = 'vc'
    results.loc[results['ventmode'] == 1, 'ventmode'] = 'pc'
    results.loc[results['ventmode'] == 3, 'ventmode'] = 'ps'
    results.loc[results['ventmode'] == 4, 'ventmode'] = 'cpap'
    results.loc[results['ventmode'] == 6, 'ventmode'] = 'pav'
    results['elapsed_time'] = results.end_time - results.start_time
    try:
        os.mkdir(args.patient_results_path)
    except OSError:
        pass
    results.to_pickle(pickle_path)


def func_star(args):
    try:
        analyze_patient(*args)
    except:
        warn('Encountered error when scanning patient {}. You should re-run this'.format(args[1]))
        with open('ventmode_error_reporting.txt', 'a') as f:
            f.write(args[1] + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vwd_base_dir')
    parser.add_argument('saved_classifier_path')
    parser.add_argument('saved_scaler_path')
    parser.add_argument('--patient-results-path', default='ventmode_results/')
    parser.add_argument('--threads', type=int)
    parser.add_argument('--skip-if-analyzed', action='store_true')
    parser.add_argument('-t', '--time-thresh', type=int, help='if ventmode time is under this amount just ignore and attribute it to larger flanking regions', default=4)
    parser.add_argument('--only-patient', help='only run analysis for a single patient')
    parser.add_argument('--debug', action='store_true', help='do not run in multiprocessing to help debug potential issues with scan')
    args = parser.parse_args()

    fileset = datasets.make_fileset_from_base_dir(args.vwd_base_dir, PATIENT_PATTERN)
    input_gen = [(args, patient_id, list(patient_fs)) for patient_id, patient_fs in groupby(fileset['x'], lambda x: x[0])]
    if args.only_patient:
        input_gen = filter(lambda x: x[1] == args.only_patient, input_gen)
    if args.debug:
        for input_ in input_gen:
            analyze_patient(*input_)
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() if not args.threads else args.threads, maxtasksperchild=1)
        pool.map(func_star, input_gen)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
