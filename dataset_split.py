import numpy as np
import pandas as pd


xing = pd.read_csv('data/XING/XING/XING.csv')

cold_xing = xing.sample(frac=0.2, random_state=1234)
warm_xing = xing.drop(cold_xing.index)

cold_xing.reset_index(inplace=True, drop=True)
warm_xing.reset_index(inplace=True, drop=True)

cold_xing_train = cold_xing.sample(frac=0.8, random_state=1234)
cold_xing_valid_test = cold_xing.drop(cold_xing_train.index)
cold_xing_valid = cold_xing_valid_test.sample(frac=0.5, random_state=1234)
cold_xing_test = cold_xing_valid_test.drop(cold_xing_valid.index)
cold_xing_train.reset_index(inplace=True, drop=True)
cold_xing_valid.reset_index(inplace=True, drop=True)
cold_xing_test.reset_index(inplace=True, drop=True)

warm_xing_train = warm_xing.sample(frac=0.8, random_state=1234)
warm_xing_valid_test = warm_xing.drop(warm_xing_train.index)
warm_xing_valid = warm_xing_valid_test.sample(frac=0.5, random_state=1234)
warm_xing_test = warm_xing_valid_test.drop(warm_xing_valid.index)
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

cold_cite = cite.sample(frac=0.2, random_state=1234)
warm_cite = cite.drop(cold_cite.index)

cold_cite.reset_index(inplace=True, drop=True)
warm_cite.reset_index(inplace=True, drop=True)

cold_cite_train = cold_cite.sample(frac=0.8)
cold_cite_valid_test = cold_cite.drop(cold_cite_train.index)
cold_cite_valid = cold_cite_valid_test.sample(frac=0.5)
cold_cite_test = cold_cite_valid_test.drop(cold_cite_valid.index)
cold_cite_train.reset_index(inplace=True, drop=True)
cold_cite_valid.reset_index(inplace=True, drop=True)
cold_cite_test.reset_index(inplace=True, drop=True)

warm_cite_train = warm_cite.sample(frac=0.8)
warm_cite_valid_test = warm_cite.drop(warm_cite_train.index)
warm_cite_valid = warm_cite_valid_test.sample(frac=0.5)
warm_cite_test = warm_cite_valid_test.drop(warm_cite_valid.index)
warm_cite_train.reset_index(inplace=True, drop=True)
warm_cite_valid.reset_index(inplace=True, drop=True)
warm_cite_test.reset_index(inplace=True, drop=True)

cold_xing_train.to_csv(path_or_buf='data/processed/cold_xing_train.csv', index=False, header=True)
cold_xing_valid.to_csv(path_or_buf='data/processed/cold_xing_valid.csv', index=False, header=True)
cold_xing_test.to_csv(path_or_buf='data/processed/cold_xing_test.csv', index=False, header=True)
cold_cite_train.to_csv(path_or_buf='data/processed/cold_cite_train.csv', index=False, header=True)
cold_cite_valid.to_csv(path_or_buf='data/processed/cold_cite_valid.csv', index=False, header=True)
cold_cite_test.to_csv(path_or_buf='data/processed/cold_cite_test.csv', index=False, header=True)

warm_xing_train.to_csv(path_or_buf='data/processed/warm_xing_train.csv', index=False, header=True)
warm_xing_valid.to_csv(path_or_buf='data/processed/warm_xing_valid.csv', index=False, header=True)
warm_xing_test.to_csv(path_or_buf='data/processed/warm_xing_test.csv', index=False, header=True)
warm_cite_train.to_csv(path_or_buf='data/processed/warm_cite_train.csv', index=False, header=True)
warm_cite_valid.to_csv(path_or_buf='data/processed/warm_cite_valid.csv', index=False, header=True)
warm_cite_test.to_csv(path_or_buf='data/processed/warm_cite_test.csv', index=False, header=True)