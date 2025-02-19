import pandas as pd
xing = pd.read_csv('./data/XING/XING/Xing.csv')
# i = 0
# for line in xing.itertuples():
#     i = i+1
#     if i >= 10:
#         break
#     print(line.item)
print(max(xing['item'])+1)