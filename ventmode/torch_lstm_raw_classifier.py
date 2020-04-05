from __future__ import print_function
import argparse

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from ventmode.torch_loader import RawDataset


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


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden=None):
        out, (hn, cn) = self.lstm(input, hidden)
        # hn gives you the output from the last lstm cell used in the packed
        # sequence calculations
        out = self.classifier(hn.squeeze(0))
        return self.softmax(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-preloaded', default='train-preprocessed-75.pkl')
    parser.add_argument('--test-preloaded', default='test-preprocessed-75.pkl')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--cuda', action='store_true', help='run on gpu')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('-lr', '--learning-rate', type=float, default=.01)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('-n', '--num-layers', type=int, default=1)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.cuda else x
    to_cpu_wrapper = lambda x: x.cpu() if args.cuda else x

    input_size = 2

    criterion = nn.CrossEntropyLoss()
    model = cuda_wrapper(LSTMClassifier(input_size, args.hidden_size, args.num_layers))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    train_loss = []

    train_dataset = RawDataset(args.train_preloaded)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    with torch.enable_grad():
        for ep in range(args.epochs):
            for idx, (seq, labels) in enumerate(train_loader):
                model.zero_grad()
                optimizer.zero_grad()
                labels = cuda_wrapper(Variable(labels.long()))
                seq = cuda_wrapper(Variable(seq))
                packed, sort_mask = pack_seq(seq)
                pred = model(packed)
                loss = criterion(pred, labels[sort_mask])
                loss.backward()
                optimizer.step()
                train_loss.append(float(to_cpu_wrapper(loss).detach().numpy()))
                print("epoch {}/{} batch {}/{}, av loss: {}\r".format(ep+1, args.epochs, idx, len(train_loader), round(np.mean(train_loss), 4)), end="")

    test_dataset = RawDataset(args.test_preloaded)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    model.eval()
    preds = None
    gt = None

    with torch.no_grad():
        for idx, (seq, labels) in enumerate(test_loader):
            labels = cuda_wrapper(Variable(labels.long()))
            seq = cuda_wrapper(Variable(seq))
            packed, sort_mask = pack_seq(seq)
            pred = model(packed)
            if preds is None:
                preds = pred
                gt = labels[sort_mask]
            else:
                preds = torch.cat((preds, pred))
                gt = torch.cat((gt, labels[sort_mask]))

    preds = preds.argmax(dim=1)
    print(classification_report(gt, preds))


if __name__ == "__main__":
    main()
