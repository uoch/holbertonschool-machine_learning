#!/usr/bin/env python3
import numpy as np
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

hg = df['High'].tail(10)
close = df['Close'].tail(10)
"""
a = np.ndarray(shape=(10, 2), dtype=float, order='F')
a[:, 0] = hg
a[:, 1] = close
"""
df_2 = pd.DataFrame({'High': hg.values, 'Close': close.values})
a = df_2.to_numpy()
print(a)
