from __future__ import print_function
import argparse

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from ventmode.torch_loader import LSTMFeaturizedDataset


class GRUNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, cuda_wrapper, seq_size):
        super(GRUNetwork, self).__init__()
        input_size = 7
        self.hidden_units = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, self.hidden_units, num_layers=self.num_layers)
        # it seems that without the linear layer the gradient isn't destroyed
        self.classifier = nn.Linear(self.hidden_units, 5)
        self.softmax = nn.Softmax(dim=2)
        self.cuda_wrapper = cuda_wrapper
        self.seq_size = seq_size

    def init_hidden(self):
        return self.cuda_wrapper(nn.Parameter(torch.zeros(self.num_layers, self.seq_size, self.hidden_units), requires_grad=True))

    def forward(self, x, hidden):
        x, hidden = self.gru(x, hidden)
        #return self.softmax(x), hidden
        return self.softmax(self.classifier(x)), hidden


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_size, num_layers, cuda_wrapper, seq_size):
        super(LSTMNetwork, self).__init__()
        input_size = 7
        self.hidden_units = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_units, num_layers=self.num_layers)
        # it seems that without the linear layer the gradient isn't destroyed
        self.classifier = nn.Linear(self.hidden_units, 5)
        self.softmax = nn.Softmax(dim=2)
        self.cuda_wrapper = cuda_wrapper
        self.seq_size = seq_size

    def init_hidden(self):
        return (self.cuda_wrapper(torch.randn(self.num_layers, self.seq_size, self.hidden_units)),
                self.cuda_wrapper(torch.randn(self.num_layers, self.seq_size, self.hidden_units)))

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        #return self.softmax(x), hidden
        return self.softmax(self.classifier(x)), hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('-t', '--type', choices=['gru', 'lstm'], default='lstm')
    # XXX batch size is a bit of a misnomer here. its more like sequence length
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--cuda', action='store_true', help='run on gpu')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('-n', '--num-layers', type=int, default=1)
    parser.add_argument('-lr', '--learning-rate', type=float, default=.001)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.cuda else x
    to_cpu_wrapper = lambda x: x.cpu() if args.cuda else x

    train_dataset = LSTMFeaturizedDataset(args.dataset_path, 'train')
    test_dataset = LSTMFeaturizedDataset(args.dataset_path, 'test')
    train_loader = DataLoader(train_dataset, seq_size=args.seq_size, shuffle=args.shuffle)
    test_loader = DataLoader(test_dataset, seq_size=args.seq_size)

    criterion = nn.CrossEntropyLoss()
    if args.type == 'gru':
        model = cuda_wrapper(GRUNetwork(args.hidden_size, args.num_layers, cuda_wrapper, args.seq_size))
    elif args.type == 'lstm':
        model = cuda_wrapper(LSTMNetwork(args.hidden_size, args.num_layers, cuda_wrapper, args.seq_size))
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    train_loss = []
    last_patient = None
    # XXX need to figure out how to segment between patients. Can probably
    # do so using collate_fn in the dataloader. Look at example in retinanet
    with torch.enable_grad():
        for ep in range(args.epochs):
            hidden = model.init_hidden()
            for idx, (seq, labels, patient) in enumerate(train_loader):
                if not args.shuffle and len(set(patient)) > 1:
                    # Because we cutoff non-whole sized batches just continue onto
                    # next batch
                    hidden = model.init_hidden()
                    continue
                elif args.shuffle:
                    hidden = model.init_hidden()
                elif args.seq_size == 1 and last_patient != patient:
                    hidden = model.init_hidden()

                labels = cuda_wrapper(Variable(labels).long())
                seq = cuda_wrapper(Variable(seq))
                bs, ts, features = seq.size()
                # cutoff batches that are not sized
                if bs < args.seq_size:
                    continue
                seq = seq.view((ts, bs, features))
                if args.type == 'lstm':
                    pred, (hx, cx) = model(seq, hidden)
                    hidden = (hx.detach(), cx.detach())
                elif args.type == 'gru':
                    pred, hx = model(seq, hidden)
                    hidden = hx.detach()
                pred = pred.squeeze(0)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()

                model.zero_grad()
                optimizer.zero_grad()
                train_loss.append(float(to_cpu_wrapper(loss).detach().numpy()))
                print("epoch {}/{} batch {}/{}, av loss: {}\r".format(ep+1, args.epochs, idx, len(train_loader), np.mean(train_loss)), end="")
                del loss

                if args.seq_size == 1:
                    last_patient = patient

    model.eval()
    preds = None
    gt = None
    hidden = model.init_hidden()

    with torch.no_grad():
        for idx, (seq, labels, patient) in enumerate(test_loader):
            if len(set(patient)) > 1:
            #if not args.shuffle and len(set(patient)) > 1:
                # Because we cutoff non-whole sized batches just continue onto
                # next batch
                hidden = model.init_hidden()
                continue

            labels = cuda_wrapper(Variable(labels).long())
            seq = cuda_wrapper(Variable(seq))
            bs, ts, features = seq.size()
            # cutoff trailing seq
            if bs < args.seq_size:
                continue
            seq = seq.view((ts, bs, features))
            pred, hidden = model(seq, hidden)
            pred = pred.squeeze(0)
            if preds is None:
                preds = pred
                gt = labels
            else:
                preds = torch.cat((preds, pred))
                gt = torch.cat((gt, labels))

    preds = preds.argmax(dim=1)
    print(classification_report(gt, preds, digits=5))


if __name__ == "__main__":
    main()
