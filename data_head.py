import numpy as np
import pandas as pd
xing = pd.read_csv('./data/XING/XING/Xing.csv')
print(np.array(xing.index))
#
# print(np.array([1, 2, 3]))