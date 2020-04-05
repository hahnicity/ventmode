"""
datasets
~~~~~~~~

Returns a raw data frame of feature data relevant to our problem

One big problem that I had on my last ML problem was versioning of
feture sets. So let's version them now and then link the versioning
in some documentation.
"""
import csv
from glob import glob
import math
import os
from os.path import basename
import random
import re
import traceback

import numpy as np
import pandas as pd
from scipy import mean, var
from ventmap.breath_meta import get_production_breath_meta
from ventmap.clear_null_bytes import clear_descriptor_null_bytes
from ventmap.detection import detect_version_v2
from ventmap.raw_utils import extract_raw
from ventmap.SAM import calc_pressure_itime_by_dyn_threshold, calc_pressure_itime_by_pip, calc_pressure_itime_from_front, check_if_plat_occurs

from ventmode.dtw_lib import dtw_file_analyze
from ventmode import constants


class V1FeatureSet(object):
    def __init__(self, fileset):
        self.fileset = fileset
        # pressure x0 modifier (seconds)
        self.px0 = .1
        # flow x0 modifier, basically count from .1 seconds
        # from max_f to .1 seconds before x0
        self.fx0 = .1
        # amount of time used to measure mean inspiratory pressure
        #
        # This is quite a bit of timw and I'm unclear why I set this up to
        # begin with. Yeah. this is just an artifact of v1 and has nothing to do
        # with the current version
        self.insp_p_to = .5
        self.dt = 0.02

        self.f_idxs = int(self.fx0 / self.dt)
        self.p_idxs = int(self.px0 / self.dt)
        self.insp_p_to_idxs = int(self.insp_p_to / self.dt)

    def determine_mode(self):
        y_array = []
        for patient, filename in self.fileset['y']:
            yfile = pd.read_csv(filename)
            try:
                yfile.simv
            except AttributeError:
                yfile['simv'] = np.nan
            try:
                yfile.pav
            except AttributeError:
                yfile['pav'] = np.nan
            for i, row in yfile.iterrows():
                if row.vc == 1:
                    mode = 0
                elif row.pc == 1:
                    mode = 1
                elif row.prvc == 1:
                    mode = 2
                elif row.ps == 1:
                    mode = 3
                elif row.cpap_sbt == 1:
                    mode = 4
                elif row.simv == 1:
                    mode = 5
                elif row.pav == 1:
                    mode = 6
                else:
                    mode = 7

                rel_bn = row.BN
                vent_bn = row[1]
                y_array.append([patient, rel_bn, vent_bn, mode])
        return y_array

    def get_features_and_annos(self):
        x_array = []
        y_array = self.determine_mode()
        for patient, filename in self.fileset['x']:
            breath_meta = get_file_breath_meta(filename)
            for breath in breath_meta[1:]:
                rel_bn = breath[0]
                vent_bn = breath[1]
                itime = breath[6]
                tvi = breath[9]
                maxf = breath[12]
                x0_idx = breath[28]
                x_array.append([filename, patient, rel_bn, vent_bn, tvi, maxf, x0_idx, itime])

        x = pd.DataFrame(x_array, columns=['x_filename', 'patient', 'rel_bn', "vent_bn", 'tvi', 'maxf', 'x0_idx', 'itime'])
        y = pd.DataFrame(y_array, columns=['patient', 'rel_bn', "vent_bn", 'y'])
        return x.merge(y, 'outer', on=['patient', 'rel_bn', 'vent_bn']).dropna()

    def interbreath_features(self, df):
        """
        Extract features from raw VWD on a per-breath basis
        """
        flow_var = []
        pressure_var = []
        insp_p_array = []
        files = df.x_filename.unique()
        for file_ in files:
            patient_df = df[df.x_filename == file_]
            bns = patient_df.vent_bn.tolist()
            generator = extract_raw_speedup(open(file_, "rb"), True, spec_vent_bns=bns)
            for b_idx, breath in enumerate(generator):
                # iloc for faster searches
                x0_idx = int(patient_df.iloc[b_idx].x0_idx)
                if x0_idx - self.f_idxs <= 0:
                    flow_var.append(np.nan)
                    pressure_var.append(np.nan)
                    insp_p_array.append(np.nan)
                    continue
                flow = breath['flow']
                pressure = breath['pressure']

                f_slice = flow[self.f_idxs:(x0_idx - self.f_idxs)]
                p_slice = pressure[self.p_idxs:(x0_idx - self.p_idxs)]
                p_slice_insp_p_to = pressure[:self.insp_p_to_idxs]

                # just drop breaths that have too few points
                if len(f_slice) < 10:
                    flow_var.append(np.nan)
                    pressure_var.append(np.nan)
                    insp_p_array.append(np.nan)
                    continue

                # calculate interbreath flow/pressure points
                f_diffs = [(f_slice[idx + 3] - f) for idx, f in enumerate(f_slice[:-3])]
                p_diffs = [(p_slice[idx + 3] - f) for idx, f in enumerate(p_slice[:-3])]
                flow_var.append(var(f_diffs))
                pressure_var.append(var(p_diffs))
                insp_p_array.append(mean(p_slice_insp_p_to))

        # naming here is slightly misleading. This is the variance of the
        # interpoint flow/pressure diffs
        df['flow_var'] = flow_var
        df['pressure_var'] = pressure_var
        # I will need to make this var name configurable if I start changing things
        df['insp_p_bs_to_{}'.format(self.insp_p_to)] = insp_p_array
        return df.dropna()

    def create_df(self):
        df = self.get_features_and_annos()
        df = self.interbreath_features(df)
        # calculate extra-breath features
        df['tvi_diff'] = df.tvi.shift(1) - df.tvi
        df['maxf_diff'] = df.maxf.shift(1) - df.maxf
        df['itime_diff'] = df.itime.shift(1) - df.itime
        # the below is to identify prvc
        df['insp_p_bs_to_{}_diff'.format(self.insp_p_to)] = df['insp_p_bs_to_{}'.format(self.insp_p_to)].shift(1) - df['insp_p_bs_to_{}'.format(self.insp_p_to)]
        df['patient'] = df.x_filename.str.extract("(\w{32})-")
        return df.dropna()


class V2FeatureSet(V1FeatureSet):

    def get_features_and_annos(self):
        x_array = []
        y_array = self.determine_mode()
        for patient, filename in self.fileset['x']:
            breath_meta = get_file_breath_meta(filename)
            for breath in breath_meta[1:]:
                rel_bn = breath[0]
                vent_bn = breath[1]
                itime = breath[6]
                tvi = breath[9]
                maxf = breath[12]
                maxp = breath[14]
                ipauc = breath[18]
                x0_idx = breath[28]
                x_array.append([filename, patient, rel_bn, vent_bn, tvi, maxf, x0_idx, itime, maxp, ipauc])

        x = pd.DataFrame(x_array, columns=['x_filename', 'patient', 'rel_bn', "vent_bn", 'tvi', 'maxf', 'x0_idx', 'itime', 'maxp', 'ipauc'])
        y = pd.DataFrame(y_array, columns=['patient', 'rel_bn', "vent_bn", 'y'])
        return x.merge(y, 'outer', on=['patient', 'rel_bn', 'vent_bn']).dropna()

    def create_df(self):
        df = self.get_features_and_annos()
        df = self.interbreath_features(df)
        # calculate extra-breath features
        df['tvi_diff'] = df.tvi.shift(1) - df.tvi
        df['maxf_diff'] = df.maxf.shift(1) - df.maxf
        df['itime_diff'] = df.itime.shift(1) - df.itime
        df['maxp_diff'] = df.maxp.shift(1) - df.maxp
        df['ipauc_diff'] = df.ipauc.shift(1) - df.ipauc
        # the below is to identify prvc
        df['insp_p_bs_to_{}_diff'.format(self.insp_p_to)] = df['insp_p_bs_to_{}'.format(self.insp_p_to)].shift(1) - df['insp_p_bs_to_{}'.format(self.insp_p_to)]
        df['patient'] = df.x_filename.str.extract("(\w{32})-")
        return df.dropna()


class VFinalFeatureSet(V1FeatureSet):

    def __init__(self, fileset, min_window_len, max_window_len, patient_pattern='(\w{32})-'):
        self.fileset = fileset
        super(VFinalFeatureSet, self).__init__(fileset)
        # .1 seems to perform best. Although in the future I
        # should really consider putting just a range of these guys into
        # a df and see which variable statistically works.
        self.insp_p_to = .1
        self.var_window = min_window_len
        # I've found 100 to be the best interval via testing
        self.var_window_max = max_window_len
        self.p_idxs_front = int(.26 / .02)
        self.p_idxs_back = int(.04 / .02)
        self.f_idxs_front = int(.06 / .02)
        self.f_idxs_back = int(.04 / .02)
        self.point_diffs = 4
        # Drop breaths if they have < 10 points in their insp flow slice
        #
        # XXX Should we allow the algorithm the flexibility to utilize missing data?
        # Want to have at least 3 values so that we can take a flow slope variance
        self.drop_len = self.point_diffs + 3
        self.patient_pattern = patient_pattern

    def extract_breath_info(self, filename, spec_rel_bns):
        with open(filename, 'rU') as f:
            f = clear_descriptor_null_bytes(f)
            first_line = f.readline()
            bs_col, ncol, ts_1st_col, ts_1st_row = detect_version_v2(first_line)
            f.seek(0)
            return list(extract_raw(f, False, spec_rel_bns=spec_rel_bns))

    def get_features(self, tor_results=None, random_thresh=None, dtw_n_lookback=None, dtw_thresh=None, dtw_rolling_av_len=None):
        x_array = []
        all_breath_vwd = dict()
        all_breath_metadata = dict()
        for patient, filename in self.fileset['x']:
            if not filename in all_breath_metadata:
                all_breath_metadata[filename] = []
            if tor_results is None:
                spec_rel_bns = []
            else:
                file_results = tor_results[tor_results.filename == filename]
                spec_rel_bns = file_results[
                    (file_results.sumt == 0) &
                    (file_results['bs.1or2'] == 0) &
                    (file_results['dbl.4'] == 0)
                ].BN.values.tolist()
            try:
                pt_vwd = self.extract_breath_info(filename, spec_rel_bns)
            except Exception as err:
                print "Error in runtime:\n\n{}\n\nskip file: {}".format(traceback.format_exc(), filename)
                continue

            if random_thresh:
                tmp_vwd = []
                for breath in pt_vwd:
                    if random.random() > random_thresh:
                        tmp_vwd.append(breath)
                pt_vwd = tmp_vwd

            if dtw_thresh:
                tmp_vwd = []
                scores, _ = dtw_file_analyze(pt_vwd, dtw_n_lookback, dtw_rolling_av_len)
                for idx, score in enumerate(scores):
                    if not score or score is np.nan or score > dtw_thresh:
                        continue
                    tmp_vwd.append(pt_vwd[idx])
                pt_vwd = tmp_vwd

            all_breath_vwd[filename] = pt_vwd
            pt_array = []
            for idx, vwd in enumerate(pt_vwd):
                if not vwd:
                    continue
                breath = get_production_breath_meta(vwd)
                all_breath_metadata[filename].append(breath)
                frame_dur = vwd['frame_dur']
                rel_bn = breath[0]
                vent_bn = breath[1]
                try:
                    abs_bs = pd.to_datetime(breath[29], format='%Y-%m-%d %H-%M-%S.%f')
                except:
                    abs_bs = pd.to_datetime(breath[29], format='%Y-%m-%d %H:%M:%S.%f')
                itime = breath[6]
                etime = breath[7]
                tvi = breath[9]
                maxf = breath[12]
                maxp = breath[14]
                pip = breath[15]
                maw = breath[16]
                peep = breath[17]
                ipauc = breath[18]
                epauc = breath[19]
                x0_idx = breath[28]
                pt_array.append({
                    'x_filename': filename,
                    'patient': patient,
                    'rel_bn': rel_bn,
                    'vent_bn': vent_bn,
                    'abs_bs': abs_bs,
                    'breath_time': frame_dur,
                    'tvi': tvi,
                    'maxf': maxf,
                    'x0_idx': x0_idx,
                    'itime': itime,
                    'etime': etime,
                    'maxp': maxp,
                    'ipauc': ipauc,
                    'epauc': epauc,
                    'ipauc:epauc': ipauc / epauc,
                    'pip': pip,
                    'maw': maw,
                    'peep': peep,
                    'pip_min_peep': pip - peep
                })

                # If we're not within the var_window, then don't collect any
                # observations for the dataset
                if idx + 1 - self.var_window < 0:
                # If we're within the var window, but not the max var window, no
                # worries, just take what we have available for the max
                # var window.
                    median_pip, median_peep, pressure_itime = np.nan, np.nan, np.nan
                    pressure_itime_dot5, pressure_itime_dot6, pressure_itime_dot65, pressure_itime_dot7 = np.nan, np.nan, np.nan, np.nan
                else:
                    min_idx = 0 if idx + 1 - self.var_window_max < 0 else idx + 1 - self.var_window_max
                    slice = pt_array[min_idx:idx+1]
                    # XXX I think this is actually pretty bugged. But somehow it works??
                    #
                    # Yea. because whatever you did with the model ended up being
                    # predictive of cpap because this is pretty high in cpap. And also
                    # because this was probably all tuned to what you wanted it to
                    # be with maw and ipauc
                    median_pip = np.median(map(lambda x: x['pip'], slice))
                    median_peep = np.median(map(lambda x: x['peep'], slice))
                    # this one has the lowest variance of all the calcs, however,
                    # for some reason cpap detection takes a dip when I use it.
                    pressure_itime = calc_pressure_itime_from_front(vwd['t'], vwd['pressure'], median_pip, median_peep, .4)
                    pressure_itime_dot5 = calc_pressure_itime_from_front(vwd['t'], vwd['pressure'], median_pip, median_peep, .5)
                    pressure_itime_dot6 = calc_pressure_itime_from_front(vwd['t'], vwd['pressure'], median_pip, median_peep, .6)
                    pressure_itime_dot7 = calc_pressure_itime_from_front(vwd['t'], vwd['pressure'], median_pip, median_peep, .7)
                    pressure_itime_dot65 = calc_pressure_itime_from_front(vwd['t'], vwd['pressure'], median_pip, median_peep, .65)
                    # this is one method of calculating pressure itime. it generally
                    # works very well and it resistant to noise on expiratory lim,
                    # however, if there is heavy noise on insp. lim this algo
                    # begins to lost a lot of precision.
                    #pressure_itime = calc_pressure_itime_by_dyn_threshold(vwd['t'], vwd['pressure'], median_pip, median_peep, .25)
                pt_array[-1].update({
                    'pressure_itime': pressure_itime,
                    'median_pip': median_pip,
                    'median_peep': median_peep,
                    'pressure_itime_.5': pressure_itime_dot5,
                    'pressure_itime_.6': pressure_itime_dot6,
                    'pressure_itime_.65': pressure_itime_dot65,
                    'pressure_itime_.7': pressure_itime_dot7,
                })

            # calculate extra-breath features
            for idx, row in enumerate(pt_array):
                min_idx = idx + 1 - self.var_window
                if min_idx < 0:
                    tvi_var = np.nan
                    itime_var = np.nan
                    maw_var = np.nan
                    pressure_itime_var = np.nan
                    pressure_itime_var_longer = np.nan
                    itime_var_longer = np.nan
                else:
                    window_max_idx = 0 if idx + 1 - self.var_window_max < 0 else idx + 1 - self.var_window_max
                    slice = pt_array[min_idx:idx+1]
                    slice_max = pt_array[window_max_idx:idx+1]
                    tvi_var = var(map(lambda x: x['tvi'], slice))
                    itime_var = var(map(lambda x: x['itime'], slice))
                    maw_var = var(map(lambda x: x['maw'], slice))
                    pressure_slice = map(lambda x: x['pressure_itime'], slice)
                    max_pressure_slice = map(lambda x: x['pressure_itime'], slice_max)
                    pressure_itime_var = var(filter(lambda x: x if x is not np.nan else None, pressure_slice))
                    pressure_itime_var_longer = var(filter(lambda x: x if x is not np.nan else None, pressure_slice))
                    itime_var_longer = var(map(lambda x: x['itime'], slice_max))
                pt_array[idx].update({
                    'tvi_var': tvi_var,
                    'itime_var': itime_var,
                    'maw_var': maw_var,
                    'pressure_itime_var': pressure_itime_var,
                    'pressure_itime_var_max_win': pressure_itime_var_longer,
                    'itime_var_max_win': itime_var_longer
                })
            x_array.extend(pt_array)
        x = pd.DataFrame(x_array)
        x = self.interbreath_features(x, all_breath_vwd)
        x = x.sort_values(by=['patient', 'abs_bs'])
        x.index = range(len(x))
        return x, all_breath_vwd, all_breath_metadata

    def get_features_and_annos(self, tor_results=None, random_thresh=None,
                               dtw_n_lookback=None, dtw_thresh=None,
                               dtw_rolling_av_len=None, reductions=None):
        """
        Get all features and y annotations for a learning dataframe
        """
        y_array = self.determine_mode()
        x, all_breath_vwd, all_breath_metadata = self.get_features(tor_results=tor_results, random_thresh=random_thresh, dtw_n_lookback=dtw_n_lookback, dtw_thresh=dtw_thresh, dtw_rolling_av_len=dtw_rolling_av_len)
        y = pd.DataFrame(y_array, columns=['patient', 'rel_bn', "vent_bn", 'y'])
        if tor_results or random_thresh or dtw_thresh:
            how = 'inner'
        else:
            how = 'outer'
        df = x.merge(y, how, on=['patient', 'rel_bn', 'vent_bn'])
        bad_merged = df[df.x_filename.isnull()].patient.unique()
        if tor_results is None and len(bad_merged) > 0:
           raise Exception("There are patients that were not matched correctly in the merge! {}".format(bad_merged))
        elif tor_results is not None:
            df = df[~df.x_filename.isnull()]

        # Need to reconcile inner merges with any dangling vwd or breath metadata
        # we didn't have annotations for
        if how == 'inner':
            for filename in all_breath_vwd:
                file_df = df[df.x_filename == filename]
                if len(file_df) != len(all_breath_vwd[filename]):
                    annotated_bns = file_df.rel_bn.values
                    new_breath_meta = []
                    new_vwd = []
                    for idx, breath in enumerate(all_breath_vwd[filename]):
                        if breath['rel_bn'] in annotated_bns:
                            new_vwd.append(breath)
                            new_breath_meta.append(all_breath_metadata[filename][idx])
                    all_breath_vwd[filename] = new_vwd
                    all_breath_metadata[filename] = new_breath_meta

        if reductions:
            # must reconstruct record of which filename is matched with which rel bn
            if dtw_thresh or random_thresh:
                first_bns = y[y.rel_bn - y.shift(1).rel_bn != 1]
                y_complete = y.merge(x, 'left', on=['patient', 'rel_bn', 'vent_bn'])
                for i, (idx, row) in enumerate(first_bns.iterrows()):
                    if i + 1 == len(first_bns):
                        end_idx = y_complete.iloc[-1].name
                    else:
                        end_idx = first_bns.iloc[i+1].name - 1
                    slice = y_complete.loc[idx:end_idx]
                    file_rows = slice[~slice.x_filename.isna()]
                    if len(file_rows.x_filename.unique()) > 1:
                        raise Exception('something went wrong with your code')
                    if len(file_rows) != 0:
                        y_complete.loc[slice.index[0]:slice.index[-1], 'x_filename'] = file_rows.x_filename.values[0]
            else:
                y_complete = df.copy()

            for filename in y_complete.x_filename.unique():
                to_discard = []
                for cls, n_breaths in reductions:
                    cls_in_file = y_complete[(y_complete.y == cls) & (y_complete.x_filename == filename)]
                    to_discard.extend(cls_in_file.rel_bn[n_breaths:].tolist())

                df = df[~((df.rel_bn.isin(to_discard)) & (df.x_filename == filename))]

                new_breath_meta = []
                new_vwd = []
                for idx, breath in enumerate(all_breath_vwd[filename]):
                    if breath['rel_bn'] not in to_discard:
                        new_vwd.append(breath)
                        new_breath_meta.append(all_breath_metadata[filename][idx])
                all_breath_vwd[filename] = new_vwd
                all_breath_metadata[filename] = new_breath_meta

        df.index = range(len(df))
        return df, all_breath_vwd, all_breath_metadata

    def interbreath_features(self, df, all_breath_vwd):
        """
        Extract features from raw VWD on a per-breath basis
        """
        flow_var = []
        all_pressure_var = []
        pressure_var = []
        x0_minus_1 = []
        flow_var_var = []
        flow_slope_mean = []
        num_plats_past_20 = []
        num_plats_past_40 = []
        files = df.x_filename.unique()
        x0_pos_in_df = list(df.columns).index('x0_idx')

        for pt_idx, file_ in enumerate(files):
            patient_df = df[df.x_filename == file_]
            patient_x0_minus_arr = []
            pt_flow_var = []
            plat_detected = []
            try:
                vwd = all_breath_vwd[file_]
            except KeyError:
                continue

            for b_idx, breath in enumerate(vwd):
                x0_idx = int(patient_df.iat[b_idx, x0_pos_in_df])
                # If there are fewer than (self.f_idxs / .02) points in insp flow
                if x0_idx - self.f_idxs <= 0:
                    pt_flow_var.append(np.nan)
                    patient_x0_minus_arr.append(np.nan)
                    all_pressure_var.append(np.nan)
                    pressure_var.append(np.nan)
                    flow_var_var.append(np.nan)
                    flow_slope_mean.append(np.nan)
                    plat_detected.append(np.nan)
                    num_plats_past_20.append(np.nan)
                    num_plats_past_40.append(np.nan)
                    continue

                flow = breath['flow']
                pressure = breath['pressure']
                patient_x0_minus_arr.append(flow[x0_idx-1])

                f_slice = flow[self.f_idxs_front:(x0_idx - self.f_idxs_back)]
                p_slice = flow[self.f_idxs_front:(x0_idx - self.f_idxs_back)]
                # just drop breaths that have too few points
                if len(f_slice) < self.drop_len:
                    pt_flow_var.append(np.nan)
                    all_pressure_var.append(np.nan)
                    pressure_var.append(np.nan)
                    flow_var_var.append(np.nan)
                    flow_slope_mean.append(np.nan)
                    plat_detected.append(np.nan)
                    num_plats_past_20.append(np.nan)
                    num_plats_past_40.append(np.nan)
                    continue

                plat_detected.append(check_if_plat_occurs(flow, pressure, self.dt, min_time=.2))
                all_pressure_var.append(var(pressure))
                # calculate interbreath flow/pressure points
                f_diffs = [(f_slice[idx + self.point_diffs] - f) for idx, f in enumerate(f_slice[:-self.point_diffs])]
                p_diffs = [(p_slice[idx + self.point_diffs] - f) for idx, f in enumerate(p_slice[:-self.point_diffs])]
                pt_flow_var.append(var(f_diffs))
                pt_flow_var_nan_filtered = []
                for f in pt_flow_var[::-1]:
                    if f is not np.nan:
                        pt_flow_var_nan_filtered.append(f)
                    if len(pt_flow_var_nan_filtered) >= self.var_window:
                        break
                flow_var_var.append(var(pt_flow_var_nan_filtered))
                flow_slope_mean.append(np.mean(f_diffs))
                pressure_var.append(var(p_diffs))
                num_plats_past_20.append(sum(map(lambda x: 1 if x is True else 0, plat_detected[-20:])))
                num_plats_past_40.append(sum(map(lambda x: 1 if x is True else 0, plat_detected[-40:])))
            x0_minus_1.extend(patient_x0_minus_arr)
            flow_var.extend(pt_flow_var)

        # naming here is slightly misleading. This is the variance of the
        # interpoint flow/pressure diffs
        df['flow_var'] = flow_var
        df['all_pressure_var'] = all_pressure_var
        df['flow_var_var'] = flow_var_var
        df['flow_slope_mean'] = flow_slope_mean
        df['n_plats_past_20'] = num_plats_past_20
        df['n_plats_past_40'] = num_plats_past_40
        df['pressure_var'] = pressure_var
        return df

    def _perform_additional_processing(self, df, all_vwd, all_bm):
        # XXX why is this method here? is it necessary?
        #df['patient'] = df.x_filename.str.extract(self.patient_pattern)
        return df.dropna()

    def create_learning_df(self):
        df, all_vwd, all_bm = self.get_features_and_annos()
        return self._perform_additional_processing(df, all_vwd, all_bm)

    def create_prediction_df(self):
        df, all_vwd, all_bm = self.get_features()
        return self._perform_additional_processing(df, all_vwd, all_bm)

    def create_df_and_filter_by_dtw(self, n_breaths_lookback, dtw_thresh, dtw_rolling_av_len):
        df, all_vwd, all_bm = self.get_features_and_annos(dtw_n_lookback=n_breaths_lookback, dtw_thresh=dtw_thresh, dtw_rolling_av_len=dtw_rolling_av_len)
        return self._perform_additional_processing(df, all_vwd, all_bm)

    def create_df_and_filter_random_breaths(self, random_thresh):
        df, all_vwd, all_bm = self.get_features_and_annos(random_thresh=random_thresh)
        return self._perform_additional_processing(df, all_vwd, all_bm)

    def create_df_and_remove_random_patients(self, patient_thresh):
        uniq_pts = list(set([pt for pt, _ in self.fileset['x']]))
        n_to_remove = int(len(uniq_pts) * patient_thresh)
        random.shuffle(uniq_pts)
        pts_to_remove = uniq_pts[:n_to_remove]
        new_x_fileset = []
        for pt, filename in self.fileset['x']:
            if pt not in pts_to_remove:
                new_x_fileset.append((pt, filename))
        new_y_fileset = []
        for pt, filename in self.fileset['y']:
            if pt not in pts_to_remove:
                new_y_fileset.append((pt, filename))
        self.fileset = {'x': new_x_fileset, 'y': new_y_fileset}
        df, all_vwd, all_bm = self.get_features_and_annos()
        return self._perform_additional_processing(df, all_vwd, all_bm)

    def create_df_with_optimal_size_reduction(self, reductions, random_thresh, dtw_thresh, dtw_n_lookback, dtw_rolling_av_len):
        df, all_vwd, all_bm = self.get_features_and_annos(random_thresh=random_thresh, dtw_n_lookback=dtw_n_lookback, dtw_thresh=dtw_thresh, dtw_rolling_av_len=dtw_rolling_av_len, reductions=reductions)
        df = self._perform_additional_processing(df, all_vwd, all_bm, False)
        return df


# Class to make dataset with Murias' method which essentially follows
# their 2016 paper.
#
# Overall we found that the results for this dataset were extremely weak
# when we performed the analysis on a per-breath basis. Machine learning
# helped the methodology but not by much
#
class Murias(VFinalFeatureSet):
    def __init__(self, fileset):
        """
        For Murias' method, we get the following variables over a 20 breath window

        1. insp time
        2. insp flow
        3. insp slope
        4. peak airway pressure (paw)
        5. number of paw levels - I dunno what this means
        6. TVi
        7. 300 msec pause between insp and exp.

        You don't have to implement #7 in this paper tho because it's only used
        for identifying PAV

        I can also probably get away with not using #5 either, because we're not
        doing PAV detection
        """
        self.fileset = fileset
        self.var_window = 20
        self.point_diffs = 4
        self.fx0 = .1
        self.f_idxs = int(self.fx0 / .02)
        self.f_idxs_front = int(.06 / .02)
        self.f_idxs_back = int(.04 / .02)
        self.drop_len = 10

    def get_features(self):
        """
        This method can handle the following features:

        1. insp time
        4. paw
        6. TVi
        """
        x_array = []
        all_breath_vwd = dict()
        all_breath_metadata = dict()
        for patient, filename in self.fileset['x']:
            if not filename in all_breath_metadata:
                all_breath_metadata[filename] = []
            try:
                pt_vwd = self.extract_breath_info(filename)
            except Exception as err:
                print "Error in runtime:\n\n{}\n\nskip file: {}".format(traceback.format_exc(), filename)
                continue
            all_breath_vwd[filename] = pt_vwd
            pt_array = []
            for idx, vwd in enumerate(pt_vwd):
                if not vwd:
                    continue
                breath = get_production_breath_meta(vwd)
                all_breath_metadata[filename].append(breath)
                frame_dur = vwd['frame_dur']
                rel_bn = breath[0]
                vent_bn = breath[1]
                abs_bs = breath[29]
                itime = breath[6]
                tvi = breath[9]
                maxf = breath[12]
                pip = breath[15]
                peep = breath[17]
                x0_idx = breath[28]
                pt_array.append([filename, patient, rel_bn, vent_bn, abs_bs, frame_dur, tvi, maxf, x0_idx, itime, pip])

            # calculate extra-breath features
            for idx, row in enumerate(pt_array):
                min_idx = idx + 1 - self.var_window
                if min_idx < 0:
                    tvi_var = np.nan
                    itime_var = np.nan
                    paw_var = np.nan
                else:
                    slice = pt_array[min_idx:idx+1]
                    # The 0, 1 calc here is
                    #
                    # sqrt((actual - mean)**2 / actual)
                    tvi_slice = map(lambda x: x[5], slice)
                    tvi_mean = sum(tvi_slice) / float(len(tvi_slice))
                    itime_slice = map(lambda x: x[8], slice)
                    itime_mean = sum(itime_slice) / float(len(itime_slice))
                    pip_slice = map(lambda x: x[9], slice)
                    pip_mean = sum(pip_slice) / float(len(pip_slice))
                    try:
                        tvi_var = math.sqrt((tvi_slice[-1] - tvi_mean) ** 2 / tvi_mean)
                        itime_var = math.sqrt((itime_slice[-1] - itime_mean) ** 2 / itime_mean)
                        paw_var = math.sqrt((pip_slice[-1] - pip_mean) ** 2 / pip_mean)
                        tvi_var = 1 if tvi_var > .1 else 0
                        itime_var = 1 if itime_var > .1 else 0
                        paw_var = 1 if paw_var > .1 else 0
                    except ValueError:
                        tvi_var = np.nan
                        itime_var = np.nan
                        paw_var = np.nan
                    # 1 for variable, 0 for const
                pt_array[idx].extend([tvi_var, itime_var, paw_var])
            x_array.extend(pt_array)
        x = pd.DataFrame(x_array, columns=[
            'x_filename', 'patient', 'rel_bn', "vent_bn", 'abs_bs', 'breath_time', 'tvi', 'maxf',
            'x0_idx', 'itime', 'pip', 'tvi_var', 'itime_var', 'paw_var'
        ])
        return x, all_breath_vwd, all_breath_metadata

    def interbreath_features(self, df, all_breath_vwd):
        """
        Extract features from raw VWD on a per-breath basis

        This handles the following features:

        2. insp flow
        3. insp slope
        """
        flow_var = []
        slope_var = []
        files = df.x_filename.unique()
        x0_pos_in_df = list(df.columns).index('x0_idx')
        for pt_idx, file_ in enumerate(files):
            patient_df = df[df.x_filename == file_]
            try:
                vwd = all_breath_vwd[file_]
            except KeyError:
                continue
            slope_slice = []
            flow_slice = []
            for b_idx, breath in enumerate(vwd):
                x0_idx = int(patient_df.iat[b_idx, x0_pos_in_df])

                # If there are fewer than (self.f_idxs / .02) points in insp flow
                if x0_idx - self.f_idxs <= 0:
                    flow_var.append(np.nan)
                    slope_var.append(np.nan)
                    continue
                flow = breath['flow']
                pressure = breath['pressure']

                f_slice = flow[self.f_idxs_front:(x0_idx - self.f_idxs_back)]
                # just drop breaths that have too few points
                if len(f_slice) < self.drop_len:
                    flow_var.append(np.nan)
                    slope_var.append(np.nan)
                    continue

                maxf_idx = np.argmax(f_slice)
                slope = (f_slice[-1] - f_slice[maxf_idx]) / ((len(f_slice) - maxf_idx) * 0.02)
                slope_slice.append(slope)
                if len(slope_slice) > self.var_window:
                    slope_slice.pop(0)

                flow_mean = sum(f_slice) / float(len(f_slice))
                try:
                    breath_flow_var = np.mean([math.sqrt((f - flow_mean) ** 2 / flow_mean) for f in f_slice])
                except ValueError:
                    flow_var.append(np.nan)
                    slope_var.append(np.nan)
                    continue

                if b_idx + 1 < self.var_window:
                    flow_var.append(np.nan)
                    slope_var.append(np.nan)
                    continue

                slope_mean = sum(slope_slice) / float(len(slope_slice))
                slope_var_tmp = math.sqrt(abs((slope_slice[-1] - slope_mean) ** 2 / slope_mean))

                slope_var.append(1 if slope_var_tmp > .1 else 0)
                flow_var.append(1 if breath_flow_var > .1 else 0)

        # naming here is slightly misleading. This is the variance of the
        # interpoint flow/pressure diffs
        df['flow_var'] = flow_var
        df['slope_var'] = slope_var
        return df


def get_patient_dirs(dir_):
    return _get_patient_dirs(dir_, "(\w{32})-")


def _get_patient_dirs(dir_, patient_pattern):
    dirs = []
    for path in os.listdir(dir_):
        if os.path.isdir(os.path.join(dir_, path)) and re.search(patient_pattern, path):
            dirs.append(os.path.join(dir_, path))
    return dirs


def get_random_patient_files_in_path(dir_):
    files = []
    for path in get_patient_dirs(dir_):
        file = get_random_file_in_dir(path)
        if file:
            files.append(file)
    return files


def get_x_random_patient_files_in_path(dir_, x):
    files = []
    for path in get_patient_dirs(dir_):
        csv_files = glob("{}/*.csv".format(path))
        random.shuffle(csv_files)
        files.extend(csv_files[:x])
    return files


def get_random_file_in_dir(dir_):
    csv_files = glob("{}/*.csv".format(dir_))
    if len(csv_files) == 0:
		return None
    return csv_files[random.randint(0, len(csv_files) - 1)]


def make_fileset_from_files(files):
    fileset = {'x': []}
    for file_ in files:
        match = re.search("(\w{32})-", file_)
        if not match:
            continue
        patient = match.groups()[0]
        fileset['x'].append((patient, file_))
    return fileset


def make_fileset_from_patient_files(files, patient):
    fileset = {'x': []}
    for file_ in files:
        fileset['x'].append((patient, file_))
    return fileset


def make_fileset_from_base_dir(base_dir, patient_pattern):
    fileset = {'x': []}
    patient_dirs = _get_patient_dirs(base_dir, patient_pattern)
    for dir_ in patient_dirs:
        patient_id = re.search(patient_pattern, dir_).groups()[0]
        files = glob(os.path.join(dir_, '*.csv'))
        for file_ in files:
            fileset['x'].append((patient_id, file_))
    return fileset


def get_cohort_fileset(x_dir, y_dir):
    fileset = {'x': [], 'y': []}

    def gather_files(dir_, key):
        patient_regex = re.compile(r'(\w{32})-')
        for filepath in glob("{}/*".format(dir_)):
            base = basename(filepath)
            patient = patient_regex.search(base).groups()[0]
            fileset[key].append((patient, filepath))

    gather_files(x_dir, "x")
    gather_files(y_dir, "y")
    return fileset


def get_derivation_cohort_fileset():
    return get_cohort_fileset(constants.DERIVATION_COHORT_X_DIR, constants.DERIVATION_COHORT_Y_DIR)


def get_validation_cohort_fileset():
    return get_cohort_fileset(constants.VALIDATION_COHORT_X_DIR, constants.VALIDATION_COHORT_Y_DIR)
