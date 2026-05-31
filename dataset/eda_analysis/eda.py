"""
Статистический анализ (EDA) входного датасета для нейросети
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../screws_dataset.csv', index_col=0)

print("Информация о данных:")
print(df.info(), "\n")

print("Описательная статистика:")
print(df.describe(), "\n")

print("Пропуски в данных:")
print(df.isnull().sum(), "\n")

continuous_features = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L', 'Q_theor', 'eps_theor', 'etha_theor']
discrete_features = ['z1', 'z2']

sns.set_style("whitegrid")

df.hist(bins=30, figsize=(15, 10), layout=(4, 3))
plt.suptitle("Распределения всех признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_distribution.png")

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(18, 6))
axes = axes.flatten()

for i, col in enumerate(continuous_features):
    sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel('')

plt.suptitle("Индивидуальные boxplot непрерывных признаков", y=1.02)
plt.tight_layout()
plt.savefig("feature_boxplot.png")

Q1 = df[['r0', 'h', 'L']].quantile(0.25)
Q3 = df[['r0', 'h', 'L']].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outlier_mask = ((df[['r0', 'h', 'L']] < lower) | (df[['r0', 'h', 'L']] > upper)).any(axis=1)
print(f"Найдено {outlier_mask.sum()} наблюдений с выбросами")

outliers_df = df[outlier_mask]
print(outliers_df.head())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x='z1', data=df, ax=axes[0])
axes[0].set_title("Частота значений z1 (число зубьев ведущего винта)")
axes[0].set_xlabel('')
axes[0].set_ylabel('')
sns.countplot(x='z2', data=df, ax=axes[1])
axes[1].set_title("Частота значений z2 (число зубьев ведомого винта)")
axes[1].set_xlabel('')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig("teeth.png")

plt.figure(figsize=(10, 8))
corr = df[continuous_features + discrete_features].corr()  # исключаем возможные строки
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Матрица корреляций")
plt.savefig("corr_matrix.png")
