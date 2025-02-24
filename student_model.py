import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

from teacher_model import bpr, BPRDataset, metrics


class student_model(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(student_model, self).__init__()
        self.U = nn.Embedding(num_users, embedding_dim)
        self.linear_user_1 = nn.Linear(1024, 512)
        self.linear_user_2 = nn.Linear(512, 256)
        self.linear_content_1 = nn.Linear(2738, 1024)
        self.linear_content_2 = nn.Linear(1024, 256)

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
    pass


def rank_distillation_loss(t_pred_i, t_pred_j, s_pred_i, s_pred_j):
    pass


def id_distillation_loss(t_pred_i, t_pred_j, s_pred_i, s_pred_j):
    pass


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

dataset_train = xing_train_dataset = BPRDataset(features=warm_xing_train, num_item=20519, train_mat=None, num_ng=1,
                                                is_train=True)
loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=8192, shuffle=True)
lr = 0.002
wd = 0.001
epochs = 50
reg = 0.001
student = student_model(num_users=106881, embedding_dim=1024)
student.cuda()
optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)

for i in range(epochs):
    student.train()
    optimizer.zero_grad()
    loss_value = 0.0
    t1 = time.time()
    loader.dataset.ng_sample()
    for user, item_i, item_j in loader:
        teacher_pred_i, teacher_pred_j = teacher(user, item_i, item_j)
        student_pred_i, student_pred_j = student(user, item_i, item_j)
        loss = (basic_loss(student_pred_i, student_pred_j) +
                rate_distillation_loss(teacher_pred_i, teacher_pred_j, student_pred_i, student_pred_j) +
                rank_distillation_loss(student_pred_i, student_pred_j, student_pred_i, student_pred_j) +
                id_distillation_loss(student_pred_i, student_pred_j, student_pred_i, student_pred_j)
                )
        loss_value = loss.cpu().item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            student.eval()
        #     recall, ndcg = metrics(student, cold_xing_valid)
        # print(f"Epoch: {i}, Recall: {recall}, NDCG: {ndcg}, Loss: {loss_value}, Time: {time.time() - t1:.2f}")