"""
Сравнение реальных и предсказанных выходных (целевых) параметров винтов.
Скрипт работал на синтетических данных
"""

import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from cvae import ConditionalVAE, loss_function
import matplotlib.pyplot as plt
import pickle

results_df = pd.read_csv('test_res.csv')
# Создаём DataFrame с предсказанными параметрами (переименовываем для удобства)
pred_geom = results_df[['A_pred', 'r1_pred', 'r2_pred', 'r_pred', 'r0_pred', 'h_pred', 'L_pred', 'z1_pred', 'z2_pred']].copy()
pred_geom.columns = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L', 'z1', 'z2']

efficiency_true = results_df['efficiency_true']
mass_flow_true = results_df['Q_true']
eps_true = results_df['eps_true']

# Сравнение
print("Сравнение гидродинамики на тестовой выборке:")
print(f"Efficiency: MAE = {np.mean(np.abs(efficiency_true - efficiency_pred)):.4f}, "
      f"RMSE = {np.sqrt(np.mean((efficiency_true - efficiency_pred)**2)):.4f}, "
      f"MAPE = {np.mean(np.abs((efficiency_true - efficiency_pred) / efficiency_true)) * 100:.2f}%")
print(f"Mass flow: MAE = {np.mean(np.abs(mass_flow_true - mass_flow_pred)):.4f}, "
      f"RMSE = {np.sqrt(np.mean((mass_flow_true - mass_flow_pred)**2)):.4f}, "
      f"MAPE = {np.mean(np.abs((mass_flow_true - mass_flow_pred) / mass_flow_true)) * 100:.2f}%")

# Графики
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(efficiency_true, efficiency_pred, alpha=0.3)
axes[0].plot([0, 1], [0, 1], 'r--')
axes[0].set_xlabel('True efficiency')
axes[0].set_ylabel('Predicted efficiency from geometry')
axes[0].set_title('Efficiency consistency')

axes[1].scatter(mass_flow_true, mass_flow_pred, alpha=0.3)
min_mf = min(mass_flow_true.min(), mass_flow_pred.min())
max_mf = max(mass_flow_true.max(), mass_flow_pred.max())
axes[1].plot([min_mf, max_mf], [min_mf, max_mf], 'r--')
axes[1].set_xlabel('True mass flow')
axes[1].set_ylabel('Predicted mass flow from geometry')
axes[1].set_title('Mass flow consistency')
plt.tight_layout()
plt.show()