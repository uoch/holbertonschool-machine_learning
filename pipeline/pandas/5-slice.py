#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
"""
nb_rows = df.shape[0]

for i in range(0, nb_rows, 60):
    print(df.iloc[i])

print(df.tail())"""
print(df.columns)

colums = ['High', 'Low', 'Close', 'Volume_(BTC)']

df_2 = df[colums].iloc[::60]

print(df_2.tail())
