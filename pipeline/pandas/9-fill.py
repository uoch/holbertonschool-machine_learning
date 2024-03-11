#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop('Weighted_Price', axis=1)

df['Close'] = df['Close'].ffill()

for column in ['High', 'Low', 'Open']:
    df[column] = df[column].fillna(df['Close'])

for column in ['Volume_(BTC)', 'Volume_(Currency)']:
    df[column] = df[column].fillna(0)

print(df.head())
print(df.tail())
