import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from teacher_model import bpr

class student_model(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(student_model, self).__init__()
        self.U = nn.Embedding(num_users, embedding_dim)
        self.linear_user_1 = nn.Linear(1024, 512)
        self.linear_user_2 = nn.Linear(512, 256)
        self.linear_content_1 = nn.Linear(2738, 1024)
        self.linear_content_2 = nn.Linear(1024, 256)

    def forward(self, user, content):
        user = self.U(user)
        user = self.linear_user_1(user)
        user = self.linear_user_2(user)
        content = self.linear_content_1(content)
        content = self.linear_content_2(content)
        pred = (user * content).sum(dim=-1)
        return pred


teacher_model = bpr(106881, 20519, 1024)
state_dict = torch.load('bpr.pth', weights_only=True)
teacher_model.load_state_dict(state_dict)
teacher_model.eval()

xing_content = np.load('data/XING/XING/item_raw_content.npy')
cold_xing_valid = pd.read_csv('data/processed/cold_xing_valid.csv')
cold_xing_test = pd.read_csv('data/processed/cold_xing_test.csv')
warm_xing_train = pd.read_csv('data/processed/warm_xing_train.csv')
warm_xing_valid = pd.read_csv('data/processed/warm_xing_valid.csv')
warm_xing_test = pd.read_csv('data/processed/warm_xing_test.csv')

lr = 0.001
wd = 0.001
student = student_model(num_users=106881, embedding_dim=1024)
optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=wd)


