import pandas as pd
import numpy as np

df = pd.read_csv('params.csv', index_col=0)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')

# некоторые дополнительные признаки для упрощения зависимости
df['r_sum'] = df['r1'] + df['r2']
df['r_ratio'] = df['r1'] / df['r2']
df['h_rel'] = df['h'] / df['A']
df['L_rel'] = df['L'] / df['A']
df['windings'] = df['L'] / df['h']
df['modul'] = 2 * df['A'] / (df['z1'] + df['z2'])

# Линейная комбинация для КПД (пример)
df['efficiency'] = (0.1 * df['r_ratio'] +
              0.05 * df['h_rel'] +
              0.02 * df['windings'] +
              0.1 * df['modul'] / 50 +
              0.2 * (df['z1'] / 8) +
              0.2 * (df['z2'] / 8) +
              0.03 * (df['A'] / 600) +
              np.random.normal(0, 0.02, len(df)))  # шум

df['efficiency'] = np.clip(df['efficiency'], 0.0, 1.0)

# Линейная комбинация для массового расхода (положительный)
df['mass_flow'] = (0.5 * df['A'] / 100 +
             0.3 * df['h'] / 1000 +
             0.1 * df['L'] / 1000 +
             0.1 * df['modul'] +
             np.random.normal(0, 0.02, len(df)))
mass_flow = np.abs(df['mass_flow']) + 1

df_relative = df.drop(['r_sum', 'r_ratio', 'h_rel', 'L_rel', 'windings', 'modul'], axis=1)

df_relative.to_csv('params_relative.csv', index=True)
print("Новый датасет сохранён в 'params_relative.csv'")
print("Размерность:", df_relative.shape)
print("Колонки:", df_relative.columns.tolist())