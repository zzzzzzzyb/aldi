import pandas as pd
import numpy as np

cite = pd.DataFrame(columns=['user', 'item'])
with open ('data/citeulike-a/users.dat') as f:
    lines = f.readlines()
    idx = 0
    for line in lines:
        items = line.strip().split(' ')
        new_user = pd.DataFrame(columns=['user', 'item'], data=[[idx, int(item)] for item in items])
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


cold_cite_valid.to_csv(path_or_buf='data/processed/cold_cite_valid.csv', index=False, header=True)
cold_cite_test.to_csv(path_or_buf='data/processed/cold_cite_test.csv', index=False, header=True)
warm_cite_train.to_csv(path_or_buf='data/processed/warm_cite_train.csv', index=False, header=True)
warm_cite_valid.to_csv(path_or_buf='data/processed/warm_cite_valid.csv', index=False, header=True)
warm_cite_test.to_csv(path_or_buf='data/processed/warm_cite_test.csv', index=False, header=True)