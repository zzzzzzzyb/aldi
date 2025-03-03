# use BPRMF as teacher model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time


class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, features, num_item, dataset=None, num_ng=2, is_train=None):
        super(BPRDataset, self).__init__()
        self.features = features
        self.num_item = num_item
        self.is_train = is_train
        self.num_ng = num_ng
        self.dataset = dataset

    def ng_sample(self):
        assert self.is_train, 'testing'
        self.features_fill = []
        u_i_group = self.features.groupby('user')
        for user in self.features['user'].tolist():
            pos_items = u_i_group.get_group(user)['item'].tolist()
            for pos_item in pos_items:
                for t in range(self.num_ng):
                    j = np.random.randint(self.num_ng)
                    while j in pos_items:
                        j = np.random.randint(self.num_item)
                    self.features_fill.append([user, pos_item, j])

    def __len__(self):
        return len(self.features) * self.num_ng if self.is_train else len(self.features)

    def __getitem__(self, index):
        features = self.features_fill
        user = features[index][0]
        item_i = features[index][1]
        item_j = features[index][2]
        return user, item_i, item_j



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


def metrics(model, data):
    items = torch.tensor(list(set(data['item'])))
    item_embedding = model.I(items.cuda()).cuda()
    u_i_group = data.groupby('user')

    def batch_iterate(df, batch_size):
        for start in range(0, len(df), batch_size):
            yield df.iloc[start:start + batch_size]

    recall_dict = {}
    for user_item in batch_iterate(data, 2048):
        users = user_item['user'].tolist()
        user_item.reset_index(drop=True, inplace=True)
        users = model.U(torch.tensor(users).cuda()).cuda()
        pred = (users @ item_embedding.transpose(0, 1))
        topks = torch.topk(pred, k=20)[1]
        pred_items = torch.take(items.cpu(), topks.cpu()).cpu().tolist()
        for i in range(len(user_item)):
            if user_item.iloc[i]['item'] in pred_items[i]:
                tmp = recall_dict.get(user_item.iloc[i]['user'], [])
                tmp.append(pred_items[i].index(user_item.iloc[i]['item']))
                recall_dict[user_item.iloc[i]['user']] = tmp

    recall = []
    ndcg = []
    for user, lsts in recall_dict.items():
        recall.append(len(lsts) / len(u_i_group.get_group(user)))
        idcg = np.reciprocal(np.log2(np.arange(len(lsts)).astype(float) + 2)).sum()
        dcg = np.reciprocal(np.log2(np.array(lsts, dtype=float) + 2)).sum()
        ndcg.append(dcg / idcg)
    return np.mean(recall), np.mean(ndcg)


if __name__ == '__main__':
    dataset = 'cite'
    warm_train, warm_valid, warm_test = load_data(dataset)
    num_users = 5551 if dataset == 'cite' else 106881
    num_items = 16980 if dataset == 'cite' else 20519
    xing_train_dataset = BPRDataset(features=warm_train, num_item=num_items, dataset=dataset, num_ng=1,
                                    is_train=True)
    xing_train_dataset_loader = torch.utils.data.DataLoader(xing_train_dataset, batch_size=8192, shuffle=True)
    reg = 0.001
    lr = 0.002
    epochs = 30
    model = bpr(num_users, num_items, 1024)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)
    for i in range(epochs):
        t1 = time.time()
        model.train()
        loss_value = 0
        xing_train_dataset_loader.dataset.ng_sample()
        for user, item_i, item_j in xing_train_dataset_loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            pred_i, pred_j = model(user, item_i, item_j)
            model.zero_grad()
            loss = -((pred_i - pred_j).sigmoid().log().sum() + 0.001 * (
                        model.U.weight.norm() + model.I.weight.norm())).cpu()
            loss_value = loss.cpu().item()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            recall, ndcg = metrics(model, warm_valid)
        print(f"Epoch: {i}, Recall: {recall}, NDCG: {ndcg}, Loss: {loss_value}, Time: {time.time() - t1:.2f}")

    recall, ndcg = metrics(model, warm_test)
    print(f"Recall: {recall}, NDCG: {ndcg}")

    torch.save(model.state_dict(), f'model/bpr_{dataset}.pth')
