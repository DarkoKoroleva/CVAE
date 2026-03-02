import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

df = pd.read_csv('params.csv', index_col=0)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')
print("Признаки после удаления phi:")
print(df.columns.tolist())

print("Информация о данных:")
print(df.info(), "\n")

print("Описательная статистика:")
print(df.describe(), "\n")

print("Пропуски в данных:")
print(df.isnull().sum(), "\n")

continuous_features = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L']
discrete_features = ['z1', 'z2']

sns.set_style("whitegrid")
plt.figure(figsize=(15, 7))

# Гистограммы для всех числовых признаков
df.hist(bins=30, figsize=(15, 7), layout=(4, 3))
plt.title("Распределения всех признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_distribution")

# Построим отдельные boxplot для каждого непрерывного признака
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 7))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel('')

# Скрыть лишние подграфики (если останутся пустые)
for j in range(len(continuous_features), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Индивидуальные boxplot непрерывных признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_boxplot")

# Определим границы для выбросов по правилу 1.5*IQR
Q1 = df[['r0', 'h', 'L']].quantile(0.25)
Q3 = df[['r0', 'h', 'L']].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Маска выбросов (хотя бы по одному признаку)
outlier_mask = ((df[['r0', 'h', 'L']] < lower) | (df[['r0', 'h', 'L']] > upper)).any(axis=1)
print(f"Найдено {outlier_mask.sum()} наблюдений с выбросами")

# Просмотр выбросов
outliers_df = df[outlier_mask]
print(outliers_df.head())

# Анализ категориальных признаков z1 и z2
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='z1', data=df, ax=axes[0])
axes[0].set_title("Частота значений z1 (число зубьев ведущего винта)")
sns.countplot(x='z2', data=df, ax=axes[1])
axes[1].set_title("Частота значений z2 (число зубьев ведомого винта)")
plt.tight_layout()
plt.savefig("teeth")

# Корреляционный анализ
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Матрица корреляций")
plt.savefig("corr_matrix")
