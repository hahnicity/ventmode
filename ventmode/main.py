"""
main
~~~~

"""
import argparse
from collections import Counter
import os
import pickle
import random

import numpy as np
import pandas as pd
import prettytable
from scipy.stats.mstats import winsorize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC

from ventmode import datasets

FEATURE_SETS = {
    "v1": [
        "tvi_diff",
        "maxf_diff",
        "pressure_var",
        "flow_var",
        "itime_diff",
        "insp_p_bs_to_0.5",
        "insp_p_bs_to_0.5_diff",
    ],
    "v2": [
        "tvi_diff",
        "maxf_diff",
        "pressure_var",
        "flow_var",
        "itime_diff",
        "insp_p_bs_to_0.5",
        "insp_p_bs_to_0.5_diff",
        "maxp_diff",
        "ipauc_diff",
    ],
    # Currently all features commented out don't seem to affect performance
    "vfinal": [
        #"itime_var_max_win",
        #"maw_var",
        "flow_var_var",
        "all_pressure_var",
        "flow_var",
        #"pip_min_peep",
        #"tvi_var",
        #"pressure_var",
        "itime_var",
        "pressure_itime_var",
        "pressure_itime_var_max_win",
        "n_plats_past_20",
        #'median_peep',
        #'median_pip',
        #'ipauc:epauc',
        #'ipauc:itime',
        #'epauc:etime',
    ],
    "murias": [
        "itime_var",
        "flow_var",
        "slope_var",
        "paw_var",
        "tvi_var",
    ],
}

class InvalidVMPatientError(Exception):
    pass


def get_tns(actual, predictions, label):
    neg = predictions[predictions != label]
    return len(neg[neg - actual.loc[neg.index] == 0])


def get_fps(actual, predictions, label):
    pos = predictions[predictions == label]
    return len(pos[pos - actual.loc[pos.index] != 0])


def get_tps(actual, predictions, label):
    pos = predictions[predictions == label]
    return len(pos[pos - actual.loc[pos.index] == 0])


def get_fns(actual, predictions, label):
    pos = actual[actual == label]
    return len(pos[pos - predictions.loc[pos.index] != 0])


def sensitivity(actual, predictions, label):
    tps = get_tps(actual, predictions, label)
    fns = get_fns(actual, predictions, label)
    if tps + fns == 0:
        return 0
    return float(tps) / (tps + fns)


def specificity(actual, predictions, label):
    """
    Also known as the true negative rate
    """
    fp = get_fps(actual, predictions, label)
    tn = get_tns(actual, predictions, label)
    if fp == 0 and tn == 0:
        return 1
    return float(tn) / (tn + fp)


def simple_split(x, y, args):
    all_patients = x.patient.unique()
    n_test_patients = int(len(all_patients) * args.test_split_ratio)
    test_patients = np.random.choice(all_patients, size=n_test_patients, replace=False)
    train_patients = set(all_patients).difference(test_patients)
    train_idx = x[x.patient.isin(train_patients)].index
    test_idx = x[x.patient.isin(test_patients)].index
    return [(x.loc[train_idx], x.loc[test_idx], y.loc[train_idx], y.loc[test_idx])]


def rf_cross_validation(x_train, y_train, n_jobs):
    params = {
        "n_estimators": np.arange(10, 80, 10),
        "criterion": ["gini", "entropy"],
        #"min_impurity_split": [1e-6, 1e-3],
        #"min_samples_leaf": [1, 2, 3, 4],
        "max_depth": range(5, 30, 5),
        "max_features": ["auto", 'log2', None],
        "min_samples_split": [2, 3, 4, 5, 10, 15],
    }
    cv = KFold(n_splits=5)
    clf = GridSearchCV(RandomForestClassifier(random_state=1), params, cv=cv, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def svm_cross_validation(x_train, y_train, n_jobs):
    params = {
        "C": [2 ** i for i in range(-5, 3)],
        "kernel": ["rbf", "sigmoid", "linear"],
    }
    cv = KFold(n_splits=5)
    # Needs a large cache size because we have a fairly large number of observations
    clf = GridSearchCV(SVC(random_state=1, cache_size=3072), params, cv=cv, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def mlp_cross_validation(x_train, y_train, n_jobs):
    params = {
        "hidden_layer_sizes": [
            [32], [64], [128], [32, 32], [32, 64], [32, 128],
            [64, 32], [64, 64], [64, 128], [128, 32], [128, 64],
            [128, 128],
        ],
        "activation": ['relu', 'tanh', 'logistic'],
        "solver": ["adam", "sgd", "lbfgs"],
        "learning_rate_init": [.1, .01, .001, .0001],
    }
    cv = KFold(n_splits=5)
    clf = GridSearchCV(MLPClassifier(random_state=1), params, cv=cv, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def log_reg_cross_validation(x_train, y_train, n_jobs):
    params = {
        "penalty": ['l2'],
        "tol": [1e-4, 1e-5, 1e-6],
        "C": range(1, 11, 3),
        "solver": ['sag', 'newton-cg', 'lbfgs'],
        "max_iter": [100, 200],
    }
    cv = KFold(n_splits=5)
    clf = GridSearchCV(LogisticRegression(random_state=1), params, cv=cv, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    print("Best params {}".format(clf.best_params_))
    return clf.best_estimator_


def cross_patient_split(x, y, args):
    def kfold_generator():
        patients = x.patient.unique()
        # Just ensure we get some kind of random distribution of patients per fold
        np.random.shuffle(patients)
        if args.lopo:
            folds = len(patients)
            patients_per_fold = 1
        else:
            folds = args.folds
            patients_per_fold = float(len(patients)) / args.folds
        for fold_idx in range(folds):
            lower_bound = int(round(fold_idx * patients_per_fold))
            upper_bound = int(round((fold_idx + 1) * patients_per_fold))
            to_use = list(patients)[lower_bound:upper_bound]
            test_index = x[x.patient.isin(to_use)].index
            train_index = x.index.difference(test_index)
            x_test, y_test = x.loc[test_index], y.loc[test_index]
            x_train, y_train  = x.loc[train_index], y.loc[train_index]
            print("Using test patients: {}".format(", ".join(to_use)))
            yield (x_train, x_test, y_train, y_test)

    if not args.lopo and args.folds > len(x.patient.unique()):
        raise Exception("Cannot have more folds than patients!")
    return kfold_generator()


def validate_split_func(x, y, args):
    x_train = x[x.set_type == "train"]
    y_train = y.loc[x_train.index]
    x_test = x[x.set_type == 'test']
    y_test = y.loc[x_test.index]
    return [(x_train, x_test, y_train, y_test)]


class Winsorizor(object):
    def __init__(self, winsorize_val):
        self.val = winsorize_val
        self.col_max_mins = {}

    def fit_transform(self, x, to_transform=[]):
        if self.val and len(x) > 0:
            if to_transform:
                col_iter = to_transform
            else:
                col_iter = x.columns
            for col in col_iter:
                if col not in x.columns:
                    continue
                vals = winsorize(x[col], limits=self.val)
                self.col_max_mins[col] = {
                    'min': vals.min(),
                    'max': vals.max(),
                }
                x[col] = vals
        return x

    def transform(self, x):
        if self.val and len(x) > 0:
            for col, max_min in self.col_max_mins.iteritems():
                x.loc[x[x[col] > max_min['max']].index, col] = max_min['max']
                x.loc[x[x[col] < max_min['min']].index, col] = max_min['min']
        return x


class IQRFilter(object):
    """
    Applies an IQR filter @ the patient level. We do this to avoid filtering
    data points that might otherwise be considered outliers because they belong
    to a different vent mode
    """
    def __init__(self):
        self.q1 = {}
        self.q3 = {}
        self.iqr = {}

    def fit_transform(self, x):
        for patient in x.patient.unique():
            x_pt = x[x.patient == patient]
            self.q1[patient] = x_pt.quantile(.25)
            self.q3[patient] = x_pt.quantile(.75)
            self.iqr[patient] = self.q3[patient] - self.q1[patient]
        return self.transform(x)

    def transform(self, x):
        all_idx = []
        for col in set(x.columns).difference({'patient'}):
            pt_idx = []
            for patient in x.patient.unique():
                pt_idx.append(set(x[
                    (x[col] >= self.q1[patient][col] - 1.5 * self.iqr[patient][col]) &
                    (x[col] <= self.q3[patient][col] + 1.5 * self.iqr[patient][col])
                ].index))
            pt_set = pt_idx[0]
            for set_ in pt_idx:
                pt_set.intersection(set_)
            all_idx.append(pt_set)

        final_idx = all_idx[0]
        for idx_set in all_idx[1:]:
            final_idx = final_idx.union(idx_set)
        return final_idx


def murias_predictor(x):
    predictions = []
    for i, row in x.iterrows():
        if row.itime_var == 1 and row.flow_var == 1 and row.slope_var == 1 and row.paw_var == 1 and row.tvi_var == 1:
            # CPAP
            predictions.append(4)
        elif row.itime_var == 0 and row.flow_var == 0 and row.slope_var == 0 and row.paw_var == 1 and row.tvi_var == 0:
            # VC-square
            predictions.append(0)
        elif row.itime_var == 0 and row.flow_var == 1 and row.slope_var == 0 and row.paw_var == 1 and row.tvi_var == 0:
            # VC-decel
            predictions.append(0)
        elif row.itime_var == 0 and row.flow_var == 1 and row.slope_var == 1 and row.paw_var == 0 and row.tvi_var == 1:
            # PC
            predictions.append(1)
        elif row.itime_var == 1 and row.flow_var == 1 and row.slope_var == 1 and row.paw_var == 0 and row.tvi_var == 1:
            # PS
            predictions.append(3)
        else:
            # Other
            predictions.append(5)
    return predictions


class Run(object):
    def __init__(self, args):
        self.args = args
        self.results = []
        self.final_results = pd.DataFrame([], columns=['cls', 'f1', 'acc', 'sen', 'spec', 'prec', 'train_len', 'test_len', 'n_train_pts', 'n_test_pts', 'tns', 'tps', 'fns', 'fps'])

    def preprocess(self, x_train, x_test, y_train, y_test):
        self.train_pt = x_train.patient
        self.test_pt = x_test.patient
        self.test_rel_bn = x_test.rel_bn  # This is for debug purposes
        x_train = x_train[FEATURE_SETS[self.args.feature_set]]
        x_test = x_test[FEATURE_SETS[self.args.feature_set]]
        if self.args.iqr_filter:
            x_train["patient"] = self.train_pt
            x_test["patient"] = self.test_pt
            print("samples before {}".format(len(x_train)))
            iqr = IQRFilter()
            train_loc = iqr.fit_transform(x_train)
            test_loc = iqr.fit_transform(x_test)
            x_train = x_train.drop("patient", axis=1)
            x_test = x_test.drop("patient", axis=1)
            x_train, y_train = x_train.loc[train_loc], y_train.loc[train_loc]
            x_test, y_test = x_test.loc[test_loc], y_test.loc[test_loc]
            print("samples after {}".format(len(x_train)))

        if self.args.winsorize:
            win = Winsorizor(self.args.winsorize)
            x_train = win.fit_transform(x_train)
            x_test = win.transform(x_test)

        x_train, x_test = self.scale(x_train, x_test)
        return x_train, x_test, y_train, y_test

    def report_results(self, df, y, y_pred, y_train, y_test):
        # report results
        table = prettytable.PrettyTable()
        table.field_names = ['label', 'f1-score', 'accuracy', 'sensitivity', 'specificity', 'precision', 'train_len', 'test_len', 'n_train_pts', 'n_test_pts']
        for label_idx, label in enumerate(sorted(y.unique())):
            tns, tps, fns, fps = [], [], [], []
            for fold in self.results:
                if label not in fold:
                    continue
                tns.append(fold[label][2])
                tps.append(fold[label][3])
                fns.append(fold[label][4])
                fps.append(fold[label][5])
            try:
                sen = round(float(sum(tps)) / (sum(tps) + sum(fns)), 3)
            except ZeroDivisionError:
                sen = 0
            try:
                spec = round(float(sum(tns)) / (sum(tns) + sum(fps)), 3)
            except ZeroDivisionError:
                spec = 0
            try:
                precision = round(float(sum(tps)) / (sum(tps) + sum(fps)), 3)
            except ZeroDivisionError:
                precision = 0
            try:
                f1 = round(2 * (precision * sen) / (precision + sen), 3)
            except ZeroDivisionError:
                f1 = 0
            try:
                acc = round(float(sum(tps) + sum(tns)) / (sum(tps) + sum(tns) + sum(fns) + sum(fps)), 3)
            except ZeroDivisionError:
                acc = 0

            train_cls_idx = y_train[y_train == label].index
            test_cls_idx = y_test[y_test == label].index
            train_pts = len(df.loc[train_cls_idx].patient.unique())
            test_pts = len(df.loc[test_cls_idx].patient.unique())
            if self.args.split_type == 'cross_pt':
                train_len = np.nan
                test_len = np.nan
                train_pts = np.nan
                test_pts = np.nan
            else:
                train_len = int(len(y_train[y_train == label]))
                test_len = int(len(y_test[y_test == label]))
            results_row = [label, f1, acc, sen, spec, precision, train_len, test_len, train_pts, test_pts]
            table.add_row(results_row)
            self.final_results.loc[label_idx] = results_row + [sum(tns), sum(tps), sum(fns), sum(fps)]

        if not self.args.no_print_results:
            print('Average F1-score: {}'.format(self.final_results.f1.mean()))
            print('Average accuracy: {}'.format(round(self.final_results.acc.mean(), 4)))
            print(table)
            if not self.args.split_type == 'cross_pt':
                print(classification_report(y_test, y_pred, digits=5))

        if self.args.plot_conf_matrix:
            import matplotlib.pyplot as plt
            conf_mat = confusion_matrix(y, y_pred)
            norm_conf = []
            for i in conf_mat:
                a = 0
                tmp_arr = []
                a = sum(i, 0)
                for j in i:
                    tmp_arr.append(float(j)/float(a))
                norm_conf.append(tmp_arr)

                fig = plt.figure()
            plt.clf()
            ax = fig.add_subplot(111)
            ax.set_aspect(1)
            res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                            interpolation='nearest')

            width, height = conf_mat.shape

            for x in xrange(width):
                for y in xrange(height):
                    ax.annotate(str(conf_mat[x][y]), xy=(y, x),
                                horizontalalignment='center',
                                verticalalignment='center')

            cb = fig.colorbar(res)
            alphabet = ['VC', 'PC', 'PS', 'CPAP']
            plt.xlabel('Predicted')
            plt.xticks(range(width), alphabet)
            plt.ylabel('Actual')
            plt.yticks(range(height), alphabet)
            plt.savefig('images/confusion_matrix.png', format='png')

    def perform_dtw_thresh_analysis(self, df, y, predictions, y_train, y_test):
        dtw_scores = pd.read_pickle(self.args.dtw_thresh_analysis)
        dtw_scores['x_filename'] = dtw_scores['x_filename'].apply(lambda x: os.path.basename(x))
        self.dtw_thresh_results = pd.DataFrame([], columns=['thresh', 'cls', 'f1', 'sen', 'spec', 'prec', 'train_len', 'test_len', 'n_train_pts', 'n_test_pts'])
        score_threshes = [
            0, 50, 100, 150, 200, 250, 300, 350, 400, 500,
            600, 700, 800, 900, 1000, 1200, 1400, 1600,
            1800, 2000, 2500, 3000
        ]
        df['x_filename'] = df['x_filename'].apply(lambda x: os.path.basename(x))
        merged = df.merge(dtw_scores, how='inner', on=['x_filename', 'rel_bn', 'patient', 'vent_bn'])
        merged = merged[~merged.dtw.isna()]
        predictions = pd.Series(predictions, index=y_test.index)
        results_idx = 0
        for thresh in score_threshes:
            thresh_idx = merged[merged.dtw >= thresh].index
            thresh_preds = predictions.loc[thresh_idx]
            thresh_y_test = y_test.loc[thresh_idx]
            for label in sorted(y.unique()):
                tns = get_tns(thresh_y_test, thresh_preds, label)
                tps = get_tps(thresh_y_test, thresh_preds, label)
                fns = get_fns(thresh_y_test, thresh_preds, label)
                fps = get_fps(thresh_y_test, thresh_preds, label)
                try:
                    sen = round(float(tps) / (tps + fns), 3)
                except ZeroDivisionError:
                    sen = 0
                try:
                    spec = round(float(tns) / (tns + fps), 3)
                except ZeroDivisionError:
                    spec = 0
                try:
                    precision = round(float(tps) / (tps + fps), 3)
                except ZeroDivisionError:
                    precision = 0
                try:
                    f1 = round(2 * (precision * sen) / (precision + sen), 3)
                except ZeroDivisionError:
                    f1 = 0

                train_cls_idx = y_train[y_train == label].index
                test_cls_idx = thresh_y_test[thresh_y_test == label].index
                train_pts = len(df.loc[train_cls_idx].patient.unique())
                test_pts = len(df.loc[test_cls_idx].patient.unique())
                if self.args.split_type != 'validation':
                    train_len = np.nan
                    test_len = np.nan
                else:
                    train_len = int(len(y_train[y_train == label]))
                    test_len = int(len(thresh_y_test[thresh_y_test == label]))
                results_row = [thresh, label, f1, sen, spec, precision, train_len, test_len, train_pts, test_pts]
                self.dtw_thresh_results.loc[results_idx] = results_row
                results_idx += 1
        if self.args.dtw_thresh_analysis_output:
            self.dtw_thresh_results.to_pickle(self.args.dtw_thresh_analysis_output)

    def get_model(self, x_train, y_train):
        if self.args.grid_search and self.args.algo == 'rf':
            model = rf_cross_validation(x_train, y_train, self.args.grid_search_jobs)
        elif self.args.grid_search and self.args.algo == 'svm':
            model = svm_cross_validation(x_train, y_train, self.args.grid_search_jobs)
        elif self.args.grid_search and self.args.algo == 'mlp':
            model = mlp_cross_validation(x_train, y_train, self.args.grid_search_jobs)
        elif self.args.grid_search and self.args.algo == 'log_reg':
            model = log_reg_cross_validation(x_train, y_train, self.args.grid_search_jobs)
        elif self.args.algo == 'rf':
            model = RandomForestClassifier(max_features='auto', n_estimators=self.args.estimators, max_depth=10, criterion='entropy', min_samples_split=3, random_state=1)
        elif self.args.algo == 'svm':
            model = SVC(kernel='rbf', C=10)
        elif self.args.algo == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=[64, 32], activation='logistic', solver='adam', learning_rate_init=.001)
        elif self.args.algo == 'log_reg':
            model = LogisticRegression(penalty='l2', C=4, max_iter=100, tol=.0001, solver='lbfgs')
        return model

    def run(self, df):
        # XXX tmp filter for prvc, other and something else, because.
        df = df.loc[df.index.difference(df[(df.y == 2) | (df.y == 5) | (df.y == 7) | (df.y == 9)].index)]
        y = df.y
        x = df.drop('y', axis=1)
        split_func = {
            "cross_pt": cross_patient_split,
            "simple": simple_split,
            "validation": validate_split_func,
        }[self.args.split_type]

        all_predictions = []
        for x_train, x_test, y_train, y_test in split_func(x, y, self.args):
            if self.args.split_type != 'validation' and self.args.only_patient and self.args.only_patient not in x_test.patient.unique():
                continue
            elif self.args.split_type == 'validation' and self.args.only_patient:
                x_test = x_test[x_test.patient == self.args.only_patient]
                y_test = y_test.loc[x_test.index]
            if self.args.only_cls and self.args.only_cls not in y_test.unique():
                continue
            self.test_patients = x_test.patient

            # Murias just uses a heuristic rule to predict vent mode so
            # no learning is performed here
            if self.args.feature_set == "murias":
                x_test = x_test[FEATURE_SETS[self.args.feature_set]]
                predictions = murias_predictor(x_test)
            # Here we perform our learning mechanism with the Random Forest
            else:
                x_train, x_test, y_train, y_test = self.preprocess(x_train, x_test, y_train, y_test)
                model = self.get_model(x_train, y_train)
                model.fit(x_train, y_train)
                if self.args.save_classifier_to:
                    pickle.dump(model, open(self.args.save_classifier_to, 'wb'))
                    pickle.dump(self.scaler, open("{}.scaler".format(self.args.save_classifier_to), 'wb'))
                if self.args.algo == 'rf' and not self.args.no_print_results:
                    print(zip(x_train.columns, model.feature_importances_))
                if len(x_test) == 0:
                    break
                predictions = model.predict(x_test)
            predictions = pd.Series(predictions, index=y_test.index)

            # Utilize secondary windowing to smooth predictions a bit. It
            # doesn't drastically change performance though.
            if self.args.with_majority_window:
                predictions = self.perform_majority_window_voting(predictions)
            elif self.args.with_conf_window:
                predictions = self.perform_confidence_window_voting(predictions)
            elif self.args.with_lookahead_window:
                predictions = self.perform_lookahead_window(predictions)
            elif self.args.with_lookahead_conf:
                predictions = self.perform_lookahead_confidence_window(predictions)

            if self.args.time_thresh_cutoff:
                predictions = merge_periods_with_low_time_thresh(predictions, self.test_patients, x.loc[x_test.index].abs_bs, pd.Timedelta(minutes=self.args.time_thresh_cutoff))

            all_predictions.extend(predictions.values)
            if not self.args.no_print_results:
                print(classification_report(y_test, predictions.values, digits=5))
            self.results.append({})
            for label in y_test.unique():
                self.results[-1][label] = [
                    sensitivity(y_test, predictions, label),
                    specificity(y_test, predictions, label),
                    get_tns(y_test, predictions, label),
                    get_tps(y_test, predictions, label),
                    get_fns(y_test, predictions, label),
                    get_fps(y_test, predictions, label),
                ]
        self.report_results(df, y, all_predictions, y_train, y_test)
        if self.args.dtw_thresh_analysis:
            self.perform_dtw_thresh_analysis(df, y, all_predictions, y_train, y_test)
        if self.args.ipython:
            import IPython; IPython.embed()

    def perform_lookahead_confidence_window(self, predictions):
        return perform_lookahead_confidence_window(predictions, self.test_patients, self.args.with_lookahead_conf, self.args.lookahead_conf_frac)

    def perform_lookahead_window(self, predictions):
        final_preds = []
        last_patient = None
        for idx, pred in predictions.iteritems():
            cur_patient = self.test_patients.loc[idx]
            if cur_patient != last_patient:
                cur_patient_start_loc = idx
                cur_patient_end_loc = self.test_patients[self.test_patients == cur_patient].iloc[-1].index
            lookback_idx = idx - self.args.with_lookahead_window if idx - self.args.with_lookahead_window >= cur_patient_start_loc else cur_patient_start_loc
            window_items = predictions.loc[lookback_idx:idx]
            val_counts = window_items.value_counts()
            if len(val_counts) > 1:
                lookahead_idx = idx + self.args.with_lookahead_window if idx + self.args.with_lookahead_window <= cur_patient_end_loc else cur_patient_end_loc
                window_items = predictions.loc[idx:lookahead_idx]
                majority_vote = window_items.value_counts().idxmax()
                final_preds.append(majority_vote)
            else:
                final_preds.append(pred)
            last_patient = cur_patient
        return pd.Series(final_preds, index=predictions.index)

    def perform_majority_window_voting(self, predictions):
        final_preds = []
        last_patient = None
        cur_patient_start_loc = None
        for idx, pred in predictions.iteritems():
            cur_patient = self.test_patients.loc[idx]
            if cur_patient != last_patient:
                cur_patient_start_loc = idx
            lookback_idx = idx - self.args.with_majority_window if idx - self.args.with_majority_window >= cur_patient_start_loc else cur_patient_start_loc
            window_items = predictions.loc[lookback_idx:idx]
            majority_vote = window_items.value_counts().idxmax()
            final_preds.append(majority_vote)
            last_patient = cur_patient
        return pd.Series(final_preds, index=predictions.index)

    def perform_confidence_window_voting(self, predictions):
        final_preds = []
        last_patient = None
        cur_patient_start_loc = None
        for idx, pred in predictions.iteritems():
            cur_patient = self.test_patients.loc[idx]
            if cur_patient != last_patient:
                cur_patient_start_loc = idx
            lookback_idx = idx - self.args.with_conf_window if idx - self.args.with_conf_window >= cur_patient_start_loc else cur_patient_start_loc
            window_items = predictions.loc[lookback_idx:idx]
            val_counts = window_items.value_counts()
            max_votes = val_counts.max()
            majority_vote = val_counts.idxmax()
            all_votes = val_counts.sum()
            if max_votes / float(all_votes) >= self.args.conf_window_frac:
                final_preds.append(majority_vote)
            else:
                final_preds.append(pred)
            last_patient = cur_patient
        return pd.Series(final_preds, index=predictions.index)

    def scale(self, x_train, x_test):
        cols = x_train.columns
        train_index = x_train.index
        test_index = x_test.index
        if self.args.scaler == "min_max":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
        x_train = self.scaler.fit_transform(x_train)
        if len(x_test) != 0:
            x_test = self.scaler.transform(x_test)
        return pd.DataFrame(x_train, index=train_index, columns=cols), pd.DataFrame(x_test, index=test_index, columns=cols)

    def reverse_transform(self, x):
        """
        A debugging method to be used when you want to reverse a scaler's
        transformation
        """
        cols = x.columns
        index = x.index
        trans = self.scaler.inverse_transform(x)
        return pd.DataFrame(trans, columns=cols, index=index)

    def visualize_results(self, cols, x_test, y_test, predictions, labels):
        """
        Visualize the results of your algorithm in a 2D scatter plot by picking
        two columns to serve as x and y. Pick a specific label to visualize.
        Then the predictions will show up as colors

        Vol ctrl: red
        PC: blue
        PRVC: green
        PS: magenta
        CPAP: cyan
        Other: yellow
        """
        import matplotlib.pyplot as plt
        if isinstance(labels, int):
            labels = [labels]
        elif labels == []:
            labels = range(6)
        zone = predictions[predictions.isin(labels)]

        x = x_test.loc[zone.index,cols[0]]
        y = x_test.loc[zone.index,cols[1]]
        color_mapping = {0: 'r', 1: 'b', 2: 'g', 3: 'm', 4: 'c', 5: 'y'}
        if len(cols) != 2:
            raise Exception("you can only pick 2 cols to use")
        colors = []
        for class_ in list(zone):
            colors.append(color_mapping[int(class_)])
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.ylim([y.min()-0.04, y.max()+0.04])
        plt.xlim([x.min()-0.04, x.max()+0.04])
        plt.scatter(x, y, c=colors, s=5)
        plt.show()


def perform_lookahead_confidence_window(predictions, test_patients, window_len, confidence_frac):
    """
    :param predictions: Predictions in a pandas series
    :param test_patients: Series of test patients corresponding with each prediction
    :param window_len: Size of lookahead window
    :param confidence_frac: Fraction of votes you need before you can redeclare a prediction
    """
    final_preds = []
    last_patient = None
    # changing this to iterate over a dataframe takes more time than if you just
    # keep the predictions and the patients separate
    #
    # Changing this to iterate over a simple patient array saves time but also
    # introduces more problems
    for idx, pred in predictions.iteritems():
        cur_patient = test_patients.loc[idx]  # takes 25.3% of time
        if cur_patient != last_patient:
            cur_patient_start_loc = idx
            cur_patient_end_loc = test_patients[test_patients == cur_patient].index[-1]
        lookback_idx = idx - window_len if idx - window_len >= cur_patient_start_loc else cur_patient_start_loc
        window_items = predictions.loc[lookback_idx:idx]  # takes 54.2% of time
        val_counts = set(window_items.values)
        if len(val_counts) > 1:
            lookahead_idx = idx + window_len if idx + window_len <= cur_patient_end_loc else cur_patient_end_loc
            window_items = predictions.loc[idx:lookahead_idx]  # takes 9.3%
            val_counts = Counter(window_items)  # takes 3.2%
            most_represented = val_counts.most_common()[0]
            majority_vote = most_represented[0]
            max_votes = most_represented[1]
            all_votes = len(window_items)
            if max_votes / float(all_votes) >= confidence_frac:
                final_preds.append(majority_vote)
            else:
                final_preds.append(pred)
        else:
            final_preds.append(pred)
        last_patient = cur_patient
    return pd.Series(final_preds, index=predictions.index)


def merge_periods_with_low_time_thresh(predictions, patients, abs_bs, time_thresh):
    """

    """
    for patient_id in patients.unique():
        patient_idx = patients[patients == patient_id].index
        patient_preds = predictions.loc[patient_idx]
        patient_abs_bs = abs_bs.loc[patient_idx]

        cur_mode = patient_preds.iloc[0]
        start_time = patient_abs_bs.iloc[0]
        start_idx = patient_preds.index[0]
        prev_mode = cur_mode

        # results are in [start time, end time, mode, start idx, end idx, elapsed time]
        results = []
        for idx, cur_mode in patient_preds.iteritems():
            if cur_mode != prev_mode:
                results.append([start_time, abs_bs.loc[last_idx], prev_mode, start_idx, last_idx])
                start_time = abs_bs.loc[idx]
                start_idx = idx
            last_idx = idx
            prev_mode = cur_mode
        else:
            results.append([start_time, abs_bs.loc[idx], cur_mode, start_idx, last_idx])

        results = pd.DataFrame(results, columns=['start_time', 'end_time', 'ventmode', 'start_idx', 'end_idx'])
        results['elapsed_time'] = results.end_time - results.start_time

        # For now just do matching where you match the period with low amounts of time
        # with the closest period above the threshold
        valid_time_frames = results[results.elapsed_time >= time_thresh]
        invalid_time_frames = results[results.elapsed_time < time_thresh]
        if len(valid_time_frames) == 0:
            raise InvalidVMPatientError('The patient has no breath data that is valid for VM analysis')

        matches = []
        for invalid_idx, invalid in invalid_time_frames.iterrows():
            cur_match = valid_time_frames.iloc[0].name
            match_time = abs((invalid.start_time - valid_time_frames.iloc[0].end_time).to_pytimedelta().total_seconds())
            for valid_idx, valid in valid_time_frames.iloc[1:].iterrows():
                tmp = abs((invalid.start_time - valid.end_time).to_pytimedelta().total_seconds())
                if tmp < match_time:
                    cur_match = valid_idx
                    match_time = tmp
            matches.append((invalid_idx, cur_match))

        for invalid_idx, valid_idx in matches:
            start_idx = invalid_time_frames.loc[invalid_idx].start_idx
            end_idx = invalid_time_frames.loc[invalid_idx].end_idx
            new_pred = valid_time_frames.loc[valid_idx].ventmode
            predictions.loc[start_idx:end_idx] = new_pred

    return predictions


def run_dataset_with_classifier(cls, scaler, dataset, feature_set):
    if len(dataset) == 0:
        return dataset
    features = FEATURE_SETS[feature_set]
    df_scaled = scaler.transform(dataset[features])
    predictions = cls.predict(df_scaled)
    dataset['predictions'] = predictions
    return dataset


def run_dataset_with_classifier_and_lookahead(cls,
                                              scaler,
                                              dataset,
                                              feature_set,
                                              window_len,
                                              confidence_frac):
    dataset = run_dataset_with_classifier(cls, scaler, dataset, feature_set)
    dataset['predictions'] = perform_lookahead_confidence_window(
        dataset.predictions, dataset.patient, window_len, confidence_frac
    )
    return dataset


def run_with_pickled_classifiers(cls_path, scaler_path, dataset_path, feature_set):
    df = pd.read_pickle(dataset_path)
    cls = pickle.load(open(cls_path))
    scaler = pickle.load(open(scaler_path))
    return run_dataset_with_classifier(cls, scaler, df, feature_set)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", choices=FEATURE_SETS.keys(), help="feature set to use", default='vfinal')
    parser.add_argument("-p", "--pickle-file", help="pickle file of the data to use")
    parser.add_argument("--to-pickle", help="pickle processed feature output")
    parser.add_argument("-mc", "--minority-classes", default=3, type=int, help="number of minority classes to resample")
    parser.add_argument("--split-type", choices=["cross_pt", "simple", 'validation'], default="validation")
    # robust for now. But eventually fix outliers.
    parser.add_argument("--scaler", choices=["min_max", "robust"], default="robust", help="which kind of scaler do you want to use?")
    parser.add_argument("--folds", default=10, type=int)
    parser.add_argument("-w", "--winsorize", type=float, help="winsorization percentage to perform on the data")
    parser.add_argument("--iqr-filter", action="store_true", help="Filter data via 1.5 * IQR criteria")
    parser.add_argument("--only-patient", help="Only run on this fold if the following patient is in it. To be used mainly for debugging.")
    parser.add_argument("--only-cls", type=int, help="Only run on this fold if the following class is in it. To be used mainly for debugging.")
    parser.add_argument('--lopo', action='store_true', help='leave one patient out cross validation')
    parser.add_argument('--min-window-len', type=int, default=10, help='minimum length of variance window')
    parser.add_argument('--max-window-len', type=int, default=100, help='maximum length of variance window')
    parser.add_argument('--plot-conf-matrix', action='store_true', help='Plot a confusion matrix of the results')
    parser.add_argument("--save-classifier-to")
    parser.add_argument('-sr', "--test-split-ratio", type=float, default=0.2, help='ratio of patients to keep in test set. E.g. .2 => 20% of patients are in test set')
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument('--estimators', type=int, default=50, help='number of trees to use in the random forest')
    parser.add_argument('--train-on-all', action='store_true', help='Convert all data in frame to training data. Good for if you want to save the classifier after learning purposes')
    parser.add_argument('--use-train-only', action='store_true', help='Only use training dataset. Good for hyperparameter tuning')

    # prediction smoothing
    parser.add_argument('--with-majority-window', type=int, help='roll a N sized sliding window over all predictions and vote based on majority wins')
    parser.add_argument('--with-conf-window', type=int, help='roll a N sized sliding window over all predictions and vote based on a confidence fraction')
    parser.add_argument('--conf-window-frac', type=float, default=.7, help='roll a N sized sliding window over all predictions and vote based on a confidence fraction')
    parser.add_argument('--with-lookahead-window', type=int, help='if unsure of prediction, then lookahead in time with N size window and use majority vote')
    parser.add_argument('--with-lookahead-conf', type=int, help='if unsure of prediction, then lookahead in time with N size window and use majority vote')
    parser.add_argument('--lookahead-conf-frac', type=float, default=.7, help='if unsure of prediction, then lookahead in time with N size window and use majority vote')

    parser.add_argument('--algo', choices=['svm', 'rf', 'log_reg', 'mlp'], default='rf')
    parser.add_argument('--train-without-pva-artifact-test-with', action='store_true')
    parser.add_argument('--no-print-results', action='store_true')
    parser.add_argument('-r', '--remove-x-frac-of-train', type=float, help='fraction of training examples to remove before testing')
    parser.add_argument('--filter-by-dtw-thresh', type=int, help='set a score threshold to filter breaths by DTW score')
    parser.add_argument('--dtw-n-lookback', type=int, default=1)
    parser.add_argument('--dtw-rolling-av-len', type=int, default=1)
    parser.add_argument('-rp', '--remove-random-patients', type=float, help="remove a certain fraction of patients from the train dataset")
    parser.add_argument('--dtw-thresh-analysis', help='Supply a dataset file with DTW scores if you want to perform an analysis of how well we predicted items over varying DTW thresholds. Results will be sent to an output location of choice')
    parser.add_argument('--dtw-thresh-analysis-output', help='output DTW thresh analysis in some file')
    parser.add_argument('--optimal-size-reduction', action='store_true', help='perform an optimal size reduction on the dataset. Can perform this action with DTW and random ablation as well')
    parser.add_argument('--time-thresh-cutoff', type=int, help='Cutoff observations that are below N number of minutes and just aggregate them with the closest observed mode')
    parser.add_argument('--ipython', action='store_true', help='drop into ipython at end of run. You need to have this installed on your system tho.')
    parser.add_argument('-gsj', '--grid-search-jobs', type=int, help='number of threads to run grid search with', default=4)
    return parser


def main():
    args = build_parser().parse_args()
    runner = Run(args)

    # only derivation cohort for now
    fileset = datasets.get_derivation_cohort_fileset()
    if not args.pickle_file:
        feature_mapping = {
            "v1": datasets.V1FeatureSet(fileset),
            "v2": datasets.V2FeatureSet(fileset),
            "vfinal": datasets.VFinalFeatureSet(fileset, args.min_window_len, args.max_window_len),
            "murias": datasets.Murias(fileset),
        }
        feature_set = feature_mapping[args.feature_set]
        if args.optimal_size_reduction:
            reduction = [(0, 450), (1, 1000), (3, 1000), (4, 70), (6, 300)]
            df = feature_set.create_df_with_optimal_size_reduction(reduction, args.remove_x_frac_of_train, args.filter_by_dtw_thresh, args.dtw_n_lookback, args.dtw_rolling_av_len)
        elif args.filter_by_dtw_thresh:
            df = feature_set.create_df_and_filter_by_dtw(args.dtw_n_lookback, args.filter_by_dtw_thresh, args.dtw_rolling_av_len)
        elif args.remove_x_frac_of_train:
            if not 0 <= args.remove_x_frac_of_train < 1:
                raise Exception('Must input a removal threshold between 0 and 1. Your input: {}'.format(args.remove_x_frac_of_train))
            df = feature_set.create_df_and_filter_random_breaths(args.remove_x_frac_of_train)
        elif args.remove_random_patients:
            df = feature_set.create_df_and_remove_random_patients(args.remove_random_patients)
        else:
            df = feature_set.create_learning_df()
        df['set_type'] = 'train'

        if args.split_type == 'validation':
            feature_set.fileset = datasets.get_validation_cohort_fileset()
            test_set = feature_set.create_learning_df()
            test_set['set_type'] = 'test'
            df = df.append(test_set)
        elif args.train_without_pva_artifact_test_with:
            feature_set.fileset = datasets.get_validation_cohort_fileset()
            test_set = feature_set.create_learning_df(False)
            test_set['set_type'] = 'test'
            df = df.append(test_set)
        # XXX this logic is bad if we want to just filter by dtw on train
        elif args.filter_by_dtw_thresh:
            feature_set.fileset = datasets.get_validation_cohort_fileset()
            test_set = feature_set.create_learning_df(False)
            test_set['set_type'] = 'test'

        if args.to_pickle:
            df.to_pickle(args.to_pickle)
    else:
        df = pd.read_pickle(args.pickle_file)

    if args.train_on_all:
        df.loc[:, 'set_type'] = 'train'

    if args.use_train_only:
        df = df[df.set_type == 'train']
    df.index = range(len(df))

    if args.train_without_pva_artifact_test_with or args.filter_by_dtw_thresh:
        args.split_type = 'validation'

    # drop infs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    runner.run(df)


if __name__ == "__main__":
    main()
