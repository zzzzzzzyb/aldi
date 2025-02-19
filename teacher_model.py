# use BPRMF as teacher model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_train=None):
        super(BPRDataset, self).__init__()
        self.features = features
        self.num_item = num_item
        self.is_train = is_train
        self.train_mat = train_mat
        self.num_ng = num_ng

    def ng_sample(self):
        assert self.is_train, 'testing'
        self.features_fill = []
        user = 0
        for x in self.features:
            i = np.random.randint(self.num_item)
            while x[i] != 1:
                i = np.random.randint(self.num_item)
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while x[j] == 1:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([user, i, j])
            user += 1

    def __len__(self):
        return len(self.features_fill) if self.is_train else len(self.features)

    def __getitem__(self, index):
        features = self.features_fill if self.is_train else self.features
        user = features[index][0]
        item_i = features[index][1]
        item_j = features[index][2] if self.is_train else features[index][1]
        return user, item_i, item_j


def convert(dataset, data: pd.DataFrame):
    if dataset == 'cite':
        res = torch.zeros(5551, 16980)
    elif dataset == 'xing':
        res = torch.zeros(106881, 20519)
    for line in data.itertuples():
        res[line.user][line.item] = 1
    return res


class bpr(nn.Module):
    def __init__(self, num_users, num_items, dimension):
        super(bpr, self).__init__()
        self.U = nn.Embedding(num_users, dimension)
        self.I = nn.Embedding(num_items, dimension)

        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.I.weight)

    def forward(self, user, item_i, item_j):
        user = self.U(user)
        item_i = self.I(item_i)
        item_j = self.I(item_j)

        pred_i = (user * item_i).sum(dim=-1)
        pred_j = (user * item_j).sum(dim=-1)
        return pred_i, pred_j


def load_data(dataset):
    if dataset == 'xing':
        warm_train = pd.read_csv('data/processed/warm_xing_train.csv')
        warm_valid = pd.read_csv('data/processed/warm_xing_valid.csv')
        warm_test = pd.read_csv('data/processed/warm_xing_test.csv')
        return warm_train, warm_valid, warm_test
    elif dataset == 'cite':
        warm_train = pd.read_csv('data/processed/warm_cite_train.csv')
        warm_valid = pd.read_csv('data/processed/warm_cite_valid.csv')
        warm_test = pd.read_csv('data/processed/warm_cite_test.csv')
        return warm_train, warm_valid, warm_test


if __name__ == '__main__':
    pass