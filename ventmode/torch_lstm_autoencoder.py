from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from ventmode.torch_loader import RawDataPreloader, RawDataset


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

        #initialize weights
        #
        # XXX Initialize more weights if we have > 1 layer
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))
        self.seq_len = seq_len

    def forward(self, input):
        encoded_input, hidden = self.lstm(input)
        unpacked, lens = pad_packed_sequence(encoded_input, batch_first=True, total_length=self.seq_len)
        encoded_input = self.relu(unpacked)
        encoded_input = pack_padded_sequence(encoded_input, lens, batch_first=True)
        return encoded_input


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, seq_len):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        #self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.seq_len = seq_len

        #initialize weights
        #
        # XXX Initialize more weights if we have > 1 layer
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        unpacked, lens = pad_packed_sequence(decoded_output, batch_first=True, total_length=self.seq_len)
        return unpacked


class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(LSTMAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, seq_len)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers, seq_len)

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)
        return decoded_output


def pack_seq(sequence):
    """
    Pack a sequence.

    returns the packed sequence and the sorting mask
    """
    # pack with batch_first
    seq_lens = [(j, len(s[s != 0]) / 2) for j, s in enumerate(sequence)]
    sorted_seq = sorted(seq_lens, key=lambda x: x[1])[::-1]
    mask = [j for j, _ in sorted_seq]
    lens = [k for _, k in sorted_seq]
    sequence = sequence[mask]
    return pack_padded_sequence(sequence, lens, batch_first=True), mask


def seq_plot(seq, ground_truth, savefig=None, show=True):
    flow = seq[:, 0]
    pressure = seq[:, 1]
    flow = (flow + RawDataPreloader.flow_mean) * RawDataPreloader.flow_std
    pressure = (pressure + RawDataPreloader.pressure_mean) * RawDataPreloader.pressure_std
    plt.plot(flow.cpu().numpy(), label='pred flow')
    plt.plot(pressure.cpu().numpy(), label='pred pressure')
    flow = ground_truth[:, 0]
    pressure = ground_truth[:, 1]
    flow = (flow + RawDataPreloader.flow_mean) * RawDataPreloader.flow_std
    pressure = (pressure + RawDataPreloader.pressure_mean) * RawDataPreloader.pressure_std
    plt.plot(flow.cpu().numpy(), label='gt flow')
    plt.plot(pressure.cpu().numpy(), label='gt pressure')
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-preloaded', default='train-preprocessed-75.pkl')
    parser.add_argument('--test-preloaded', default='test-preprocessed-75.pkl')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--cuda', action='store_true', help='run on gpu')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=.1)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('-n', '--num-layers', type=int, default=1)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.cuda else x
    to_cpu_wrapper = lambda x: x.cpu() if args.cuda else x

    input_size = 2
    seq_len = 75

    train_dataset = RawDataset(args.train_preloaded)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    criterion = nn.MSELoss()
    model = cuda_wrapper(LSTMAE(input_size, args.hidden_size, args.num_layers, seq_len))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    train_loss = []

    with torch.enable_grad():
        for ep in range(args.epochs):
            for idx, (seq, labels) in enumerate(train_loader):
                model.zero_grad()
                optimizer.zero_grad()
                labels = cuda_wrapper(Variable(labels))
                seq = cuda_wrapper(Variable(seq))
                packed, sort_mask = pack_seq(seq)
                pred = model(packed)
                loss = criterion(pred, seq[sort_mask])
                loss.backward()
                optimizer.step()
                train_loss.append(float(to_cpu_wrapper(loss).detach().numpy()))
                print("epoch {}/{} batch {}/{}, av loss: {}\r".format(ep+1, args.epochs, idx, len(train_loader), np.mean(train_loss)), end="")

                del loss
            seq_plot(pred[-1].detach(), seq[sort_mask][-1].detach(), savefig='epoch-{}-fig.png'.format(ep+1), show=False)

    model.eval()
    preds = None
    gt = None
    test_dataset = RawDataset(args.test_preloaded)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    with torch.no_grad():
        for idx, (seq, labels) in enumerate(test_loader):
            labels = cuda_wrapper(Variable(labels).long())
            seq = cuda_wrapper(Variable(seq))
            packed, sort_mask = pack_seq(seq)
            pred = model(packed)
            if preds is None:
                preds = pred
                gt = seq[sort_mask]
            else:
                preds = torch.cat((preds, pred))
                gt = torch.cat((gt, seq[sort_mask]))

    print("\ntest loss: {}".format(criterion(preds, gt)))


if __name__ == "__main__":
    main()
