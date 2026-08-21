import os
import json
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



signal = [0 for i in range(5000)]
df = pd.DataFrame()
df['signal'] = signal
df.loc[1000:1700] = 50
df.loc[2500:2800] = 50
df.loc[3000:4500] = 50
        
def generate_signal(start, end, width, height, up):
    for i in range(start, end, width):
        if up:
            up = False
            df.loc[i:i+width] = height + 5
        else:
            up = True
            df.loc[i:i+width] = height - 5

generate_signal(1100, 1490, 30, 50, up=True)
generate_signal(2600, 2750, 30, 50, up=True)
generate_signal(3100, 3430, 30, 50, up=True)
generate_signal(3830, 4160, 30, 50, up=True)

generate_signal(1190, 1250, 60, 50, up=False)
generate_signal(1370, 1430, 60, 50, up=False)
generate_signal(3920, 3980, 60, 50, up=False)


df['signal_changed'] = ((df['signal'] != 0) & (df['signal'].diff() > 0)).astype(int)
df['change_num'] = df['signal_changed'].cumsum().astype(int)
df['change_count_width'] = df['signal_changed'].rolling((2*60), center=False).sum()
df['change_count_width_reverse'] = df[::-1]['signal_changed'].rolling((2*60), center=False).sum()[::-1]


df['signal'] = ((df['change_count_width'] > 0) & (df['change_count_width_reverse'] > 0)).astype(int)
df['signal_num'] = (df['signal'].astype(int).diff() == 1).cumsum() 


df['high_point'] = df[df['signal'] == 1].groupby('signal_num').transform('max')['rpm']
df['low_point'] = df[df['signal'] == 1].groupby('signal_num').transform('min')['rpm']


df['index'] = df.index
df['change_num_index'] = np.where((df['change_num'].diff() != 0), df['index'], np.nan)
df['change_num_index'].ffill(inplace=True)
df['change_num_index_diff'] = df['change_num_index'].diff()
df['signal_incremental_length'] = np.where((df['change_num_index_diff'] >= 30) & 
                            (df['change_num_index_diff'] <= 60), df['change_num_index_diff'], 0)
