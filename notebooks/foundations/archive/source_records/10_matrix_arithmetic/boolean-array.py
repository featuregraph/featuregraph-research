import numpy as np
import pandas as pd

import tensorflow as tf

a = np.zeros((150*375), dtype=bool)
a[np.random.choice(150*375, 10000, replace=False)] = 1
a = a.reshape(150, 375)

columns = [f'material_{i}' for i in range(375)]
df = pd.DataFrame(a, columns=columns)

a[8, :].sum()

a[:, 8].sum()

df.loc[8].sum()

df['material_8'].sum()

np.where(a[:, 8] | a[:, 9]  | a[:, 15] , 1, 0).sum()

np.where(df['material_8'] | df['material_9'] | df['material_15'], 1, 0).sum()

# how many ingredients do products 15 and 20 have in common
# how many products do ingredients 20 and 25 have in common

np.where(df['material_20'] & df['material_25'], 1, 0).sum()

np.where(a[:, 20] & a[:, 25], 1, 0).sum()

np.where(a[15, : ] & a[20, :], 1, 0).sum()

np.where(df.loc[15] & df.loc[20], 1, 0).sum()
