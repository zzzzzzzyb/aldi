import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from bpr_teacher_model import bpr


class ALDIDataset(torch.utils.data.Dataset):
    def __init__(self, features, num_item, train_mat=None, num_ng=2, is_train=None, item_content=None):
        super(ALDIDataset, self).__init__()
        self.features = features
        self.num_item = num_item
        self.is_train = is_train
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.item_content = item_content

    def ng_sample(self):
        assert self.is_train, 'testing'
        self.features_fill = []
        u_i_group = self.features.groupby('user')
        for user in self.features['user'].tolist():
            pos_items = u_i_group.get_group(user)['item'].tolist()
            for pos_item in pos_items:
                for t in range(self.num_ng):
                    j = np.random.randint(self.num_item)
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
        content_i = self.item_content[item_i]
        content_j = self.item_content[item_j]
        return user, item_i, item_j, content_i, content_j


class student_model(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(student_model, self).__init__()
        self.U = nn.Embedding(num_users, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            # nn.Sigmoid(),
        )
        self.content_mlp = nn.Sequential(
            nn.Linear(2738, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 256),
            # nn.Sigmoid(),
        )

        def init_param(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                m.requires_grad = True

        for layer in self.user_mlp:
            layer.apply(init_param)

        for layer in self.content_mlp:
            layer.apply(init_param)

    def forward(self, user, content_i, content_j):
        user = self.U(user)
        user = self.user_mlp(user)

        content_i = self.content_mlp(content_i)
        content_j = self.content_mlp(content_j)
        pred_i = (user * content_i).sum(dim=-1)
        pred_j = (user * content_j).sum(dim=-1)
        return pred_i, pred_j


def metrics(model, data, content):
    items = list(set(data['item']))
    item_contents = content[items]
    item_contents = torch.tensor(item_contents).cuda()
    content_embeddings = model.content_mlp(item_contents)
    u_i_group = data.groupby('user')
    items = torch.tensor(items)

    def batch_iterate(df, batch_size):
        for start in range(0, len(df), batch_size):
            yield df.iloc[start:start + batch_size]

    recall_dict = {}
    for user_item in batch_iterate(data, 2048):
        users = user_item['user'].tolist()
        user_item.reset_index(drop=True, inplace=True)
        with torch.no_grad():
            users = model.U(torch.tensor(users).cuda()).cuda()
            users = model.user_mlp(users)
        pred = users @ content_embeddings.transpose(0, 1)
        topks = torch.topk(pred, k=20)[1]
        pred_items = torch.take(items, topks.cpu()).cpu().tolist()
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

    xing_content = np.load('data/XING/XING/item_raw_content.npy')
    cold_xing_valid = pd.read_csv('data/processed/cold_xing_valid.csv')
    cold_xing_test = pd.read_csv('data/processed/cold_xing_test.csv')
    warm_xing_train = pd.read_csv('data/processed/warm_xing_train.csv')
    warm_xing_valid = pd.read_csv('data/processed/warm_xing_valid.csv')
    warm_xing_test = pd.read_csv('data/processed/warm_xing_test.csv')


    teacher = bpr(106881, 20519, 1024)
    state_dict = torch.load('model/bpr.pth', weights_only=True)
    teacher.load_state_dict(state_dict)
    teacher.cuda()
    teacher.eval()



    dataset_train = ALDIDataset(features=warm_xing_train, num_item=20519, train_mat=None, num_ng=1,
                                is_train=True, item_content=xing_content)
    loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=8192, shuffle=True)
    lr = 0.001
    wd = 0.001
    epochs = 60
    reg = 0.001
    omega = -4
    student = student_model(num_users=106881, embedding_dim=1024)
    student.cuda()
    student.U.weight.data.copy_(teacher.U.weight.data)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, weight_decay=wd)
    i_u_group = warm_xing_train.groupby('item')
    i_map_u = [0] * 20519
    for i in list(i_u_group.groups.keys()):
        i_map_u[i] = len(i_u_group.get_group(i))
    N = np.sum(i_map_u) / 16415
    paras = student.user_mlp[0].weight
    for i in range(epochs):
        student.train()
        optimizer.zero_grad()
        t1 = time.time()
        loss_value = []
        loader.dataset.ng_sample()
        for user, item_i, item_j, content_i, content_j in loader:
            user = user.cuda()
            item_i = item_i.cuda()
            item_j = item_j.cuda()
            content_i = content_i.cuda()
            content_j = content_j.cuda()
            teacher_pred_i, teacher_pred_j = teacher(user, item_i, item_j)
            student_pred_i, student_pred_j = student(user, content_i, content_j)
            N_i = torch.tensor(i_map_u)[item_i.cpu().tolist()]
            ratio = N_i / N
            w = 2 * np.reciprocal(1 + np.exp(omega * ratio)) - 1
            w = w.cuda()
            with torch.no_grad():
                item_i = teacher.I(item_i).cuda()
                item_j = teacher.I(item_j).cuda()

            ct1 = nn.LogSigmoid()
            loss1 = -ct1(student_pred_i - student_pred_j).mean()

            loss2 = (teacher_pred_i - student_pred_i).abs().mean() + (teacher_pred_j - student_pred_j).abs().mean()

            para1 = (teacher_pred_i - teacher_pred_j)
            para2 = (student_pred_i - student_pred_j)
            ct = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=w)
            loss3 = ct(para1, para2.sigmoid())

            t_dist_j = item_j.mean(dim=0)
            t_dist_i = (item_i * (item_i - t_dist_j)).sum(dim=-1)
            s_dist_j = student.content_mlp(content_j).mean(dim=0)
            map_i = student.content_mlp(content_i)
            s_dist_i = (map_i * (map_i - s_dist_j)).sum(dim=-1) * 0.98
            criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=w)
            loss4 = criterion(t_dist_i, s_dist_i.sigmoid())

            # print(loss1, loss2, loss3, loss4)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            loss_value.append(loss.item())
            optimizer.step()
        with torch.no_grad():
            student.eval()
            recall, ndcg = metrics(student, cold_xing_valid, xing_content)
        print(f"Epoch: {i}, Recall: {recall}, NDCG: {ndcg}, Loss: {np.mean(loss_value)}, Time: {time.time() - t1:.2f}")

    recall, ndcg = metrics(student, cold_xing_test, xing_content)
    print(f"Recall: {recall}, NDCG: {ndcg}")

    torch.save(student.state_dict(), 'model/student_bpr.pth')
