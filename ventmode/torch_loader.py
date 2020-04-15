import argparse
import cPickle
from glob import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.autograd import Variable
# XXX
from torch.nn.utils.rnn import *
from torch.utils.data import Dataset
from ventmap.breath_meta import get_production_breath_meta
from ventmap.raw_utils import extract_raw

from ventmode.main import FEATURE_SETS


class FeaturizedDataset(Dataset):
    def __init__(self, pickled_dataset_filename, set_type):
        """
        :param pickled_dataset_filename: Filename for pickled dataset
        :param set_type: string for 'train' or 'test' set
        """
        df = pd.read_pickle(pickled_dataset_filename)
        df.index = range(len(df))
        # For now only want to train/test on VC (0), PC (1), PS (3), CPAP (4), and PAV (6)
        # XXX in future consider test prvc recognition
        df = df[df.y.isin([0, 1, 3, 4, 6])]
        # map to new numbers to make simpler for now
        df.loc[df[df.y == 3].index, 'y'] = 2
        df.loc[df[df.y == 4].index, 'y'] = 3
        df.loc[df[df.y == 6].index, 'y'] = 4
        train_set = df[df.set_type == 'train']
        train_patients = train_set.patient.tolist()
        test_set = df[df.set_type == 'test']
        test_patients = test_set.patient.tolist()

        y_train = train_set.y.values
        y_test = test_set.y.values
        feature_set = FEATURE_SETS['vfinal']
        scaler = RobustScaler()
        train_set = scaler.fit_transform(train_set[feature_set].values)

        if set_type == 'train':
            self.dataset = train_set
            self.y = y_train
            self.patients = train_patients
        elif set_type == 'test':
            self.dataset = scaler.transform(test_set[feature_set].values)
            self.y = y_test
            self.patients = test_patients

    def __getitem__(self, index):
        patient = self.patients[index]
        obs = self.dataset[index]
        y = self.y[index]
        # this is adapted for CE loss
        return torch.FloatTensor(obs), y, patient

    def __len__(self):
        return len(self.dataset)


class LSTMFeaturizedDataset(FeaturizedDataset):
    def __getitem__(self, index):
        patient = self.patients[index]
        obs = [self.dataset[index]]
        y = self.y[index]
        # this is adapted for CE loss
        return torch.FloatTensor(obs), y, patient


class RawDataPreloader(object):
    flow_std = 24.82785862786254
    pressure_std = 7.080962183225576
    # perform de-mean after set var to 1
    flow_mean = -0.004134245289354331
    pressure_mean = 1.7247804526208201

    def __init__(self, set_type, seq_size, output_filename):
        """
        :param set_type: string for 'train' or 'test' set
        """
        derivation_raw_dir = os.path.join(os.path.dirname(__file__), "../train_data/y_dir/")
        validation_raw_dir = os.path.join(os.path.dirname(__file__), "../test_data/y_dir/")
        if set_type == 'train':
            self.gt = self.get_ground_truth(derivation_raw_dir)
        elif set_type == 'test':
            self.gt = self.get_ground_truth(validation_raw_dir)
        self.output_filename = output_filename
        self.seq_size = seq_size

    def preload(self):
        breath_array = []
        for cur_file in self.gt.x_filename.unique():
            file_df = self.gt[self.gt.x_filename == cur_file]
            bn_interval = [file_df.iloc[0].BN, file_df.iloc[-1].BN]
            generator = extract_raw(open(cur_file), True, rel_bn_interval=bn_interval)
            idx_start = file_df.index[0]
            for idx_offset, breath in enumerate(generator):
                cur_row = self.gt.loc[idx_start + idx_offset]
                metadata = get_production_breath_meta(breath)
                x0 = metadata[28]
                # Take x0 + 10 points
                if len(breath['flow']) >= self.seq_size:
                    if x0 + 10 <= self.seq_size:
                        padding = [0] * (self.seq_size - (x0+10))
                        seq_index = x0 + 10
                    elif x0 + 10 > self.seq_size:
                        padding = []
                        seq_index = self.seq_size
                elif len(breath['flow']) < self.seq_size:
                    padding = [0] * (self.seq_size - len(breath['flow']))
                    seq_index = len(breath['flow'])

                flow = np.append((np.array(breath['flow'][:seq_index]) / self.flow_std) - self.flow_mean, padding)
                if len(flow) != self.seq_size:
                    raise Exception('this shouldnt happen')
                pressure = np.append((np.array(breath['pressure'][:seq_index]) / self.pressure_std) - self.pressure_mean, padding)
                breath_array.append((flow, pressure, cur_row.y))

        with open(self.output_filename, 'w') as f:
            cPickle.dump(breath_array, f)

    def get_ground_truth(self, dir_path):
        files = glob(dir_path + '*.csv')
        gt = pd.read_csv(files[0])
        gt['y_filename'] = files[0]
        for f in files[1:]:
            tmp = pd.read_csv(f)
            tmp['y_filename'] = f
            gt = gt.append(tmp)
        gt.index = range(len(gt))
        # main column which is a problem is ventBN
        if 'ventBN' in gt.columns:
            gt.loc[gt['ventBN'].dropna().index, 'vent BN'] = gt['ventBN'].dropna()
            gt = gt.drop(['ventBN'], axis=1)

        # XXX I need to save the x filename!
        x_dir = dir_path.replace('y_dir', 'raw_vwd')
        x_files = glob(x_dir + "*.csv")
        for yfile in gt.y_filename.unique():
            match = re.search(r'(\d{4}(?:[-_]\d{2}){5})', yfile)
            if not match:
                raise Exception('y file: {} did not match search pattern!'.format(yfile))
            pat = match.groups()[0]
            pat = pat.replace('_', '-')
            for xfile in x_files:
                if pat in xfile:
                    gt.loc[gt[gt.y_filename == yfile].index, 'x_filename'] = xfile
                    break
            else:
                raise Exception('Unable to find matching file for pattern: {}'.format(pat))
        gt['y'] = np.nan
        gt.loc[gt[gt.vc == 1].index, 'y'] = 0
        gt.loc[gt[gt.pc == 1].index, 'y'] = 1
        gt.loc[gt[gt.ps == 1].index, 'y'] = 2
        gt.loc[gt[gt.cpap_sbt == 1].index, 'y'] = 3
        gt.loc[gt[gt.pav == 1].index, 'y'] = 4
        return gt.loc[gt.y.dropna().index]


class RawDataset(Dataset):
    def __init__(self, input_file):
        """
        :param input_file: Preprocessed input file
        """
        with open(input_file) as f:
            self.data = cPickle.load(f)

    def __getitem__(self, index):
        flow, pressure, label = self.data[index]
        seq_len = len(flow)
        arr = np.append(flow.reshape((seq_len, 1)), pressure.reshape((seq_len, 1)), axis=1)
        return torch.FloatTensor(arr), label

    def __len__(self):
        return len(self.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_output_file')
    parser.add_argument('test_output_file')
    parser.add_argument('-s', '--seq-size', type=int, help='sequence size for raw data inputs', default=128)
    args = parser.parse_args()
    train = RawDataPreloader('train', args.seq_size, args.train_output_file)
    train.preload()
    test = RawDataPreloader('test', args.seq_size, args.test_output_file)
    test.preload()


if __name__ == "__main__":
    # perform pre-loading on for raw dataset.
    main()
