import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных (индекс уже есть, как в прошлом скрипте)
df = pd.read_csv('screws_dataset.csv', index_col=0)

print("Информация о данных:")
print(df.info(), "\n")

print("Описательная статистика:")
print(df.describe(), "\n")

print("Пропуски в данных:")
print(df.isnull().sum(), "\n")

# Список признаков, которые реально есть в данных
continuous_features = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L', 'Q_theor', 'eps_theor', 'etha_theor']
discrete_features = ['z1', 'z2']

sns.set_style("whitegrid")

# Гистограммы для всех числовых признаков (автоматически подстроит сетку)
df.hist(bins=30, figsize=(15, 10), layout=(4, 3))
plt.suptitle("Распределения всех признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_distribution.png")

# Boxplot – теперь используем сетку 2x5 или 3x4 (10 графиков)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 6))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel('')

plt.suptitle("Индивидуальные boxplot непрерывных признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_boxplot.png")

# Выбросы для r0, h, L (по‑прежнему работают)
Q1 = df[['r0', 'h', 'L']].quantile(0.25)
Q3 = df[['r0', 'h', 'L']].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_mask = ((df[['r0', 'h', 'L']] < lower) | (df[['r0', 'h', 'L']] > upper)).any(axis=1)
print(f"Найдено {outlier_mask.sum()} наблюдений с выбросами")

outliers_df = df[outlier_mask]
print(outliers_df.head())

# Анализ категориальных признаков z1 и z2
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='z1', data=df, ax=axes[0])
axes[0].set_title("Частота значений z1 (число зубьев ведущего винта)")
sns.countplot(x='z2', data=df, ax=axes[1])
axes[1].set_title("Частота значений z2 (число зубьев ведомого винта)")
plt.tight_layout()
plt.savefig("teeth.png")

# Корреляционная матрица (только числовые столбцы)
plt.figure(figsize=(10, 8))
corr = df[continuous_features + discrete_features].corr()  # исключаем возможные строки
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Матрица корреляций")
plt.savefig("corr_matrix.png")