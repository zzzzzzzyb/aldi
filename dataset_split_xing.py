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

cold_xing_valid.to_csv(path_or_buf='data/processed/cold_xing_valid.csv', index=False, header=True)
cold_xing_test.to_csv(path_or_buf='data/processed/cold_xing_test.csv', index=False, header=True)


warm_xing_train.to_csv(path_or_buf='data/processed/warm_xing_train.csv', index=False, header=True)
warm_xing_valid.to_csv(path_or_buf='data/processed/warm_xing_valid.csv', index=False, header=True)
warm_xing_test.to_csv(path_or_buf='data/processed/warm_xing_test.csv', index=False, header=True)
