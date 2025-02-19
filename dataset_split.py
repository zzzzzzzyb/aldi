import numpy as np
import pandas as pd


xing = pd.read_csv('data/XING/XING/XING.csv')

np.random.seed(4789)

cold_xing_idx = np.random.choice(np.arange(20519), round(0.2*20519), replace=False)
cold_xing = xing[xing['item'].isin(cold_xing_idx)]
warm_xing = xing.drop(cold_xing.index)

cold_xing.reset_index(inplace=True, drop=True)
warm_xing.reset_index(inplace=True, drop=True)

cold_xing_valid = cold_xing.sample(frac=0.5, random_state=1234)
cold_xing_test = cold_xing.drop(cold_xing_valid.index)
cold_xing_valid.reset_index(inplace=True, drop=True)
cold_xing_test.reset_index(inplace=True, drop=True)

warm_xing_train = warm_xing.sample(frac=0.8, random_state=1234)
warm_xing_valid_test = warm_xing.drop(warm_xing_train.index)
warm_xing_valid = warm_xing_valid_test.sample(frac=0.5, random_state=1234)
warm_xing_test = warm_xing_valid_test.drop(warm_xing_valid.index)
print(len(warm_xing_train))

warm_xing_user_set = set(warm_xing_train['user'])
idx_to_move = warm_xing_valid[True ^ warm_xing_valid['user'].isin(warm_xing_user_set)].index
warm_xing_train = pd.concat([warm_xing_train, warm_xing_valid.loc[idx_to_move]])
warm_xing_valid.drop(idx_to_move)
print(len(warm_xing_train))
warm_xing_item_set = set(warm_xing_train['item'])
idx_to_move = warm_xing_valid[True ^ warm_xing_valid['item'].isin(warm_xing_item_set)].index
warm_xing_train = pd.concat([warm_xing_train, warm_xing_valid.loc[idx_to_move]])
warm_xing_valid.drop(idx_to_move)
print(len(warm_xing_train))
warm_xing_user_set = set(warm_xing_train['user'])
idx_to_move = warm_xing_test[True ^ warm_xing_test['user'].isin(warm_xing_user_set)].index
warm_xing_train = pd.concat([warm_xing_train, warm_xing_test.loc[idx_to_move]])
warm_xing_test.drop(idx_to_move)
print(len(warm_xing_train))
warm_xing_item_set = set(warm_xing_train['item'])
idx_to_move = warm_xing_test[True ^ warm_xing_test['item'].isin(warm_xing_item_set)].index
warm_xing_train = pd.concat([warm_xing_train, warm_xing_test.loc[idx_to_move]])
warm_xing_test.drop(idx_to_move)
print(len(warm_xing_train))
warm_xing_train.reset_index(inplace=True, drop=True)
warm_xing_valid.reset_index(inplace=True, drop=True)
warm_xing_test.reset_index(inplace=True, drop=True)

cite = pd.DataFrame(columns=['user', 'item'])
with open ('data/citeulike-a/users.dat') as f:
    lines = f.readlines()
    idx = 0
    for line in lines:
        items = line.strip().split(' ')
        new_user = pd.DataFrame(columns=['user', 'item'], data=[[idx, item] for item in items])
        cite = pd.concat([cite, new_user], ignore_index=True)
        idx += 1

cold_cite_idx = np.random.choice(np.arange(16980), round(0.2*16980), replace=False)
cold_cite = cite[cite['item'].isin(cold_cite_idx)]
warm_cite = cite.drop(cold_cite.index)

cold_cite.reset_index(inplace=True, drop=True)
warm_cite.reset_index(inplace=True, drop=True)

cold_cite_valid = cold_cite.sample(frac=0.5)
cold_cite_test = cold_cite.drop(cold_cite_valid.index)
cold_cite_valid.reset_index(inplace=True, drop=True)
cold_cite_test.reset_index(inplace=True, drop=True)

warm_cite_train = warm_cite.sample(frac=0.8)
warm_cite_valid_test = warm_cite.drop(warm_cite_train.index)
warm_cite_valid = warm_cite_valid_test.sample(frac=0.5)
warm_cite_test = warm_cite_valid_test.drop(warm_cite_valid.index)

warm_cite_user_set = set(warm_cite_train['user'])
idx_to_move = warm_cite_valid[True ^ warm_cite_valid['user'].isin(warm_cite_user_set)].index
warm_cite_train = pd.concat([warm_cite_train, warm_cite_valid.loc[idx_to_move]])
warm_cite_valid.drop(idx_to_move)

warm_cite_item_set = set(warm_cite_train['item'])
idx_to_move = warm_cite_valid[True ^ warm_cite_valid['item'].isin(warm_cite_item_set)].index
warm_cite_train = pd.concat([warm_cite_train, warm_cite_valid.loc[idx_to_move]])
warm_cite_valid.drop(idx_to_move)

warm_cite_user_set = set(warm_cite_train['user'])
idx_to_move = warm_cite_test[True ^ warm_cite_test['user'].isin(warm_cite_user_set)].index
warm_cite_train = pd.concat([warm_cite_train, warm_cite_test.loc[idx_to_move]])
warm_cite_test.drop(idx_to_move)

warm_cite_item_set = set(warm_cite_train['item'])
idx_to_move = warm_cite_test[True ^ warm_cite_test['user'].isin(warm_cite_item_set)].index
warm_cite_train = pd.concat([warm_cite_train, warm_cite_test.loc[idx_to_move]])
warm_cite_test.drop(idx_to_move)

warm_cite_train.reset_index(inplace=True, drop=True)
warm_cite_valid.reset_index(inplace=True, drop=True)
warm_cite_test.reset_index(inplace=True, drop=True)

cold_xing_valid.to_csv(path_or_buf='data/processed/cold_xing_valid.csv', index=False, header=True)
cold_xing_test.to_csv(path_or_buf='data/processed/cold_xing_test.csv', index=False, header=True)
cold_cite_valid.to_csv(path_or_buf='data/processed/cold_cite_valid.csv', index=False, header=True)
cold_cite_test.to_csv(path_or_buf='data/processed/cold_cite_test.csv', index=False, header=True)

warm_xing_train.to_csv(path_or_buf='data/processed/warm_xing_train.csv', index=False, header=True)
warm_xing_valid.to_csv(path_or_buf='data/processed/warm_xing_valid.csv', index=False, header=True)
warm_xing_test.to_csv(path_or_buf='data/processed/warm_xing_test.csv', index=False, header=True)
warm_cite_train.to_csv(path_or_buf='data/processed/warm_cite_train.csv', index=False, header=True)
warm_cite_valid.to_csv(path_or_buf='data/processed/warm_cite_valid.csv', index=False, header=True)
warm_cite_test.to_csv(path_or_buf='data/processed/warm_cite_test.csv', index=False, header=True)