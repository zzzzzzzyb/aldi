# import torch
# import numpy as np
# import torch.nn as nn
# from teacher_model import BPRDataset
# import pandas as pd
# class bpr(nn.Module):
#     def __init__(self, num_users, num_items, dimension):
#         super(bpr, self).__init__()
#         self.U = nn.Embedding(num_users, dimension)
#         self.I = nn.Embedding(num_items, dimension)
#
#         nn.init.xavier_uniform_(self.U.weight)
#         nn.init.xavier_uniform_(self.I.weight)
#
#     def forward(self, user, item_i, item_j):
#         user = self.U(user)
#         item_i = self.I(item_i)
#         item_j = self.I(item_j)
#
#         pred_i = (user * item_i).sum(dim=-1)
#         pred_j = (user * item_j).sum(dim=-1)
#         return pred_i, pred_j
#
# a = bpr(num_users=106881, num_items=20519, dimension=2738)
# # print(a(torch.tensor([0, 1, 2, 2, 0, 1]), torch.tensor([0, 1, 2, 2, 0, 1]), torch.tensor([2, 0, 1, 0, 1, 2])))
# warm_xing_valid = pd.read_csv('data/processed/warm_xing_valid.csv')
# xing_valid_dataset = BPRDataset(features=warm_xing_valid, num_item=20519, train_mat=None, is_train=False)
# xing_valid_dataset_loader = torch.utils.data.DataLoader(xing_valid_dataset, batch_size=2048, shuffle=False)
# for user, item_i, item_j in xing_valid_dataset_loader:
#     print(user.shape, item_i.shape, item_j.shape)
#     print(a(user, item_i, item_j)[0].shape)
#     break
import math

import numpy as np
import pandas as pd
# def batch_iterate(df, batch_size):
#     # 使用 range() 生成每个批次的起始索引
#     for start in range(0, len(df), batch_size):
#         # 使用 iloc 切片返回当前批次
#         yield df.iloc[start:start+batch_size]
#
#
# df = pd.DataFrame({
#     'A': range(10),
#     'B': range(10,20),
# })
# for i in batch_iterate(df, 3):
#     for j in i.index:
#         print(j)
#
# lsts = [0, 2, 3]
# idcg = np.reciprocal(np.log2(np.arange(len(lsts)).astype(float)+2)).sum()
# dcg = np.reciprocal(np.log2(np.array(lsts, dtype=float) + 2)).sum()
# print(idcg)
df = pd.read_csv('data/processed/warm_xing_valid.csv')
a = df.groupby('user')
for user, group in a:
    print(int(user))
    print(group)
    print(type(group['item'].tolist()[0]))
    break
# for i in df.itertuples():
#     print(i.user)
#     break
# print(df)
# print(len(df.groupby('user').get_group(96385)))
# print(len(df.groupby('user').groups[96385]))
# print(len(df.groupby('item')))
# for i in df.groupby('user').groups.keys():
#     print(len(df.groupby('user').groups[i]))

# print(list(df.groupby('item').groups.keys()))

# dict = {
#     'A': 1,
#     'B': 2,
# }
#
# print(dict[['A', 'B']])
# print(np.mean(list(dict.values())))
# import torch
# # #
# a = torch.tensor([[1,2,3,4],
#                   [5,6,7,8],
#                   [9,10,11,12]], dtype=torch.float)
# b = torch.tensor([[2,3,4,5],
#                   [6,7,8,9],
#                   [10,11,12,13]], dtype=torch.float)
# print(torch.nn.functional.pairwise_distance(a,b).shape)

# print(len(a))
# print((a / a.sum(dim=1, keepdim=True))[0])


# def metrics(model, data, content):
#     items = list(set(data['item']))
#     item_contents = content[items]


# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 2
# y.backward()
# print(y.grad)

# state_dict = torch.load("model/bpr_xing.pth")
# print(state_dict)