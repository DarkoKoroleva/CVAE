import pandas as pd
import numpy as np

df = pd.read_csv('params.csv', index_col=0)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')

df['r1_ratio'] = df['r1'] / df['A']
df['r2_ratio'] = df['r2'] / df['A']
df['r_ratio']  = df['r'] / df['A']
df['r0_ratio'] = df['r0'] / df['A']
df['h_log'] = np.log1p(df['h'])
df['L_log'] = np.log1p(df['L'])

df['r1_r2_ratio'] = df['r1'] / df['r2']

df['i'] = df['z2'] / df['z1']                     # передаточное отношение
df['windings'] = df['L'] / df['h']                 # число витков

df_relative = df.drop(['r1', 'r2', 'r', 'r0'], axis=1)

df_relative.to_csv('params_relative.csv', index=True)
print("Новый датасет сохранён в 'params_relative.csv'")
print("Размерность:", df_relative.shape)
print("Колонки:", df_relative.columns.tolist())