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

from ventmode.torch_loader import FeaturizedDataset


class MLPNetwork(nn.Module):
    def __init__(self):
        super(MLPNetwork, self).__init__()
        input_size = 7
        output_size = 5
        hidden_units = 64
        self.hidden = nn.Linear(input_size, hidden_units, bias=True)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(hidden_units, output_size, bias=True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.softmax(self.classifier(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--max-batches', type=int, help='good for debugging')
    args = parser.parse_args()

    train_dataset = FeaturizedDataset(args.dataset_path, 'train')
    test_dataset = FeaturizedDataset(args.dataset_path, 'test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = MLPNetwork()
    optimizer = Adam(model.parameters(), lr=.01)
    model.train()
    train_loss = []

    # XXX need to figure out how to segment between patients. Can probably
    # do so using collate_fn in the dataloader. Look at example in retinanet
    with torch.enable_grad():
        for ep in range(args.epochs):
            for idx, (seq, labels) in enumerate(train_loader):
                model.zero_grad()
                labels = Variable(labels).long()
                seq = Variable(seq)
                pred = model(seq)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                train_loss.append(float(loss.detach().numpy()))
                print("epoch {}/{} batch {}/{}, av loss: {}\r".format(ep+1, args.epochs, idx, len(train_loader), np.mean(train_loss)), end="")
                if args.max_batches and idx >= args.max_batches:
                    break
                del loss

    model.eval()
    preds = None
    gt = None
    with torch.no_grad():
        for idx, (seq, labels) in enumerate(test_loader):
            labels = Variable(labels)
            seq = Variable(seq)
            pred = model(seq)
            if preds is None:
                preds = pred
                gt = labels
            else:
                preds = torch.cat((preds, pred))
                gt = torch.cat((gt, labels))

    preds = preds.argmax(dim=1)
    print(classification_report(gt, preds))


if __name__ == "__main__":
    main()
