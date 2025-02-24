import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from teacher_model import bpr, metrics


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
        content_i = self.item_content[item_i]
        content_j = self.item_content[item_j]
        return user, item_i, item_j, content_i, content_j


class student_model(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(student_model, self).__init__()
        self.U = nn.Embedding(num_users, embedding_dim)
        self.linear_user_1 = nn.Linear(1024, 512)
        self.linear_user_2 = nn.Linear(512, 256)
        self.linear_content_1 = nn.Linear(2738, 1024)
        self.linear_content_2 = nn.Linear(1024, 256)

        nn.init.xavier_uniform_(self.linear_user_1.weight)
        nn.init.xavier_uniform_(self.linear_user_2.weight)
        nn.init.xavier_uniform_(self.linear_content_1.weight)
        nn.init.xavier_uniform_(self.linear_content_2.weight)

    def forward(self, user, content_i, content_j):
        user = self.U(user)
        user = self.linear_user_1(user)
        user = self.linear_user_2(user)
        content_i = self.linear_content_1(content_i)
        content_i = self.linear_content_2(content_i)
        content_j = self.linear_content_1(content_j)
        content_j = self.linear_content_2(content_j)
        pred_i = (user * content_i).sum(dim=-1)
        pred_j = (user * content_j).sum(dim=-1)
        return pred_i, pred_j


def basic_loss(pred_i, pred_j):
    return -(pred_i - pred_j).sigmoid().log().sum().cpu()


def rate_distillation_loss(t_pred_i, t_pred_j, s_pred_i, s_pred_j):
    loss = ((t_pred_i - s_pred_i).pow(2).mean() + (t_pred_j - s_pred_j).pow(2).mean()).cpu()
    return loss


def rank_distillation_loss(t_pred_i, t_pred_j, s_pred_i, s_pred_j):
    loss = -(((t_pred_i - t_pred_j).sigmoid() * (s_pred_i - s_pred_j).sigmoid().log() +
              (1 - (t_pred_i - t_pred_j).sigmoid()) * (1 - (s_pred_i - s_pred_j).sigmoid).log()
              )).cpu()
    return loss


def id_distillation_loss(t_pred_i, t_pred_j, model, content_i, content_j):
    t_dist_j = t_pred_j.mean()
    t_dist_i = (t_pred_i * (t_pred_i - t_dist_j)).sum(dim=-1).sigmoid()
    with torch.no_grad():
        s_dist_j = model.linear_content_2(model.linear_content_1(content_j)).mean()
        map_i = model.linear_content_2(model.linear_content_1(content_i))
        s_dist_i = (map_i * (map_i - s_dist_j)).sum(dim=-1).sigmoid()
    loss = ((t_dist_i * s_dist_i.log()) + ((1 - t_dist_i) * (1 - s_dist_i)).log()).sum().cpu()
    return -loss


teacher = bpr(106881, 20519, 1024)
state_dict = torch.load('bpr.pth', weights_only=True)
teacher.load_state_dict(state_dict)
teacher.cuda()
teacher.eval()

xing_content = np.load('data/XING/XING/item_raw_content.npy')
cold_xing_valid = pd.read_csv('data/processed/cold_xing_valid.csv')
cold_xing_test = pd.read_csv('data/processed/cold_xing_test.csv')
warm_xing_train = pd.read_csv('data/processed/warm_xing_train.csv')
warm_xing_valid = pd.read_csv('data/processed/warm_xing_valid.csv')
warm_xing_test = pd.read_csv('data/processed/warm_xing_test.csv')

dataset_train = xing_train_dataset = ALDIDataset(features=warm_xing_train, num_item=20519, train_mat=None, num_ng=1,
                                                 is_train=True, item_content=xing_content)
loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=8192, shuffle=True)
lr = 0.001
wd = 0.001
epochs = 50
reg = 0.001
student = student_model(num_users=106881, embedding_dim=1024)
student.cuda()
student.U.weight.data.copy_(teacher.U.weight.data)
optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)
for i in range(epochs):
    student.train()
    optimizer.zero_grad()
    t1 = time.time()
    loader.dataset.ng_sample()
    for user, item_i, item_j, content_i, content_j in loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()
        content_i = content_i.cuda()
        content_j = content_j.cuda()
        teacher_pred_i, teacher_pred_j = teacher(user, item_i, item_j)
        student_pred_i, student_pred_j = student(user, content_i, content_j, student)
        loss = (basic_loss(student_pred_i, student_pred_j) +
                rate_distillation_loss(teacher_pred_i, teacher_pred_j, student_pred_i, student_pred_j) +
                rank_distillation_loss(student_pred_i, student_pred_j, student_pred_i, student_pred_j) +
                id_distillation_loss(student_pred_i, student_pred_j, student, content_i, content_j)
                )
        loss_value = loss.cpu().item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            student.eval()
        #     recall, ndcg = metrics(student, cold_xing_valid)
        # print(f"Epoch: {i}, Recall: {recall}, NDCG: {ndcg}, Loss: {loss_value}, Time: {time.time() - t1:.2f}")
