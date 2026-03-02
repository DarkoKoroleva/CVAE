import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Загрузка нового датасета
df = pd.read_csv('params_relative.csv', index_col=0)

print("\nРазмерность:", df.shape)
print("\nКолонки:", df.columns.tolist())
print("\nОписательная статистика:")
print(df.describe())

rel_features = ['r1_ratio', 'r2_ratio', 'r_ratio', 'r0_ratio', 'h', 'L', 'r1_r2_ratio','h_log','L_log']
phys_features = ['i', 'windings']
other = ['z1', 'z2', 'A']

# Гистограммы для относительных признаков
fig, axes = plt.subplots(3, 3, figsize=(15, 7))
axes = axes.flatten()
for i, col in enumerate(rel_features):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Histogram of {col}')
for j in range(len(rel_features), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Распределения относительных признаков", y=1.02)
plt.tight_layout()
plt.show()

# Boxplot для относительных признаков
fig, axes = plt.subplots(3, 3, figsize=(12, 7))
axes = axes.flatten()
for i, col in enumerate(rel_features):
    axes[i].boxplot(df[col])
    axes[i].set_title(f'Boxplot of {col}')
    axes[i].set_ylabel('')
for j in range(len(rel_features), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Индивидуальные boxplot относительных признаков", y=1.02)
plt.tight_layout()
plt.show()

# Корреляционная матрица
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Корреляционная матрица всех признаков')
plt.show()
