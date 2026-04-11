"""
Перебирает параметры Loss-функции, обучая нейросеть на них.
Выводит характеристики моделей для выбора наилучших параметров Loss-функции.
"""


import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cvae import ConditionalVAE, loss_function
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import pickle


# =====================================================
# 1. Загрузка и подготовка данных (копия из cvae.py)
# =====================================================
df = pd.read_csv('../temp/params.csv', index_col=0)
df['z1'] = df['z1'].astype(int)
df['z2'] = df['z2'].astype(int)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')

geom_cols = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L', 'z1', 'z2']
hydro_cols = ['efficiency', 'mass_flow', 'eps_theor']
data = df[geom_cols].values

hydro_data = np.column_stack([df['efficiency'], df['mass_flow'], df['eps_theor']])

scaler_geom = StandardScaler()
scaler_hydro = StandardScaler()

X = df[geom_cols].values
y = df[hydro_cols].values

# Разделение
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_scaled = scaler_geom.fit_transform(X_train)
X_val_scaled = scaler_geom.transform(X_val)
X_test_scaled = scaler_geom.transform(X_test)

y_train_scaled = scaler_hydro.fit_transform(y_train)
y_val_scaled = scaler_hydro.transform(y_val)
y_test_scaled = scaler_hydro.transform(y_test)

cont_mean = torch.FloatTensor(scaler_geom.mean_[:7])
cont_scale = torch.FloatTensor(scaler_geom.scale_[:7])

z1_train = torch.LongTensor(X_train[:, 7].astype(int) - 4)
z2_train = torch.LongTensor(X_train[:, 8].astype(int) - 4)
z1_val = torch.LongTensor(X_val[:, 7].astype(int) - 4)
z2_val = torch.LongTensor(X_val[:, 8].astype(int) - 4)
z1_test = torch.LongTensor(X_test[:, 7].astype(int) - 4)
z2_test = torch.LongTensor(X_test[:, 8].astype(int) - 4)

X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(X_train_t, y_train_t, z1_train, z2_train)
val_dataset = TensorDataset(X_val_t, y_val_t, z1_val, z2_val)
test_dataset = TensorDataset(X_test_t, y_test_t, z1_test, z2_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes_z1 = np.unique(z1_train.numpy())
weights_z1 = compute_class_weight('balanced', classes=classes_z1, y=z1_train.numpy())
class_weights_z1 = torch.FloatTensor(weights_z1)

classes_z2 = np.unique(z2_train.numpy())
weights_z2 = compute_class_weight('balanced', classes=classes_z2, y=z2_train.numpy())
class_weights_z2 = torch.FloatTensor(weights_z2)

class_weights_z1 = class_weights_z1 ** 0.5
class_weights_z2 = class_weights_z2 ** 0.5

# =====================================================
# 3. Перебор гиперпараметров
# =====================================================
betas = [0.1, 1.0, 10.0]
lambda_negs = [0.1, 0.5, 1.0]
lambda_sums = [0.1, 1.0, 10.0]
lamda_ces = [0.1, 0.5, 1.0]

os.makedirs('analysis', exist_ok=True)
results_csv = 'analysis/summary.csv'

# Загружаем предыдущие результаты, если есть
if os.path.exists(results_csv):
    results_df = pd.read_csv(results_csv)
else:
    results_df = pd.DataFrame(columns=['beta', 'lambda_neg', 'lambda_sum',
                                        'test_loss', 'test_mse', 'test_ce', 'test_neg', 'test_sum',
                                        'acc_z1', 'acc_z2'])

epochs_per_exp = 30  # можно изменить

# Фиксируем seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)

for beta, lambda_neg, lambda_sum, lambda_ce in itertools.product(betas, lambda_negs, lambda_sums, lamda_ces):
    print(f"\n=== beta={beta}, lambda_neg={lambda_neg}, lambda_sum={lambda_sum}, lamda_ce={lambda_ce} ===")

    model = ConditionalVAE(geom_dim=9, hydro_dim=2, latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    mse_losses = []
    ce_losses = []
    kl_losses = []
    neg_penalties = []
    sum_penalties = []

    for epoch in range(epochs_per_exp):
        model.train()
        total_loss = 0
        total_mse = 0
        total_ce = 0
        total_kl = 0
        total_neg = 0
        total_sum = 0
        for batch in train_loader:
            geom, hydro, z1t, z2t = batch
            optimizer.zero_grad()
            cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
            cont_target = geom[:, :7]
            loss, mse, ce, kl, neg, sum_pen = loss_function(
                cont_pred, logits_z1, logits_z2,
                cont_target, z1t, z2t,
                mean, logvar,
                cont_mean, cont_scale,
                beta=beta, lambda_neg=lambda_neg, lambda_sum=lambda_sum, lambda_ce=lambda_ce,
                weight_z1 = class_weights_z1,
                weight_z2 = class_weights_z2
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse += mse.item()
            total_ce += ce.item()
            total_kl += kl.item()
            total_neg += neg.item()
            total_sum += sum_pen.item()

        avg_loss = total_loss / len(train_dataset)
        avg_mse = total_mse / len(train_dataset)
        avg_ce = total_ce / len(train_dataset)
        avg_kl = total_kl / len(train_dataset)
        avg_neg = total_neg / len(train_dataset)
        avg_sum = total_sum / len(train_dataset)

        train_losses.append(avg_loss)
        mse_losses.append(avg_mse)
        ce_losses.append(avg_ce)
        kl_losses.append(avg_kl)
        neg_penalties.append(avg_neg)
        sum_penalties.append(avg_sum)

    # Сохранение графика
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'beta={beta}, lambda_neg={lambda_neg}, lambda_sum={lambda_sum}, lamda_ce={lambda_ce}', fontsize=14)

    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title('Total Loss')

    plt.subplot(2, 3, 2)
    plt.plot(mse_losses)
    plt.title('MSE')

    plt.subplot(2, 3, 3)
    plt.plot(ce_losses)
    plt.title('Cross-Entropy')

    plt.subplot(2, 3, 4)
    plt.plot(kl_losses)
    plt.title('KL')

    plt.subplot(2, 3, 5)
    plt.plot(neg_penalties)
    plt.title('Negative Penalty')

    plt.subplot(2, 3, 6)
    plt.plot(sum_penalties)
    plt.title('Sum Penalty')

    plt.tight_layout()
    plot_filename = f'analysis/beta{beta}_lneg{lambda_neg}_lsum{lambda_sum}_lce{lambda_ce}.png'
    plt.savefig(plot_filename)
    plt.close()

    # Тестирование
    model.eval()
    total_test_loss = 0
    mse_total = 0
    ce_total = 0
    neg_total = 0
    sum_total = 0
    correct_z1 = 0
    correct_z2 = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            geom, hydro, z1t, z2t = batch
            cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
            cont_target = geom[:, :7]
            loss, mse, ce, kl, neg, sum_pen = loss_function(
                cont_pred, logits_z1, logits_z2,
                cont_target, z1t, z2t,
                mean, logvar,
                cont_mean, cont_scale,
                beta=beta, lambda_neg=lambda_neg, lambda_sum=lambda_sum
            )
            total_test_loss += loss.item()
            mse_total += mse.item()
            ce_total += ce.item()
            neg_total += neg.item()
            sum_total += sum_pen.item()

            pred_z1 = torch.argmax(logits_z1, dim=1)
            pred_z2 = torch.argmax(logits_z2, dim=1)
            correct_z1 += (pred_z1 == z1t).sum().item()
            correct_z2 += (pred_z2 == z2t).sum().item()
            total_samples += z1t.size(0)

    avg_test_loss = total_test_loss / len(test_dataset)
    avg_mse_test = mse_total / len(test_dataset)
    avg_ce_test = ce_total / len(test_dataset)
    avg_neg_test = neg_total / len(test_dataset)
    avg_sum_test = sum_total / len(test_dataset)
    acc_z1 = correct_z1 / total_samples
    acc_z2 = correct_z2 / total_samples

    # Сохраняем результаты
    new_row = pd.DataFrame({
        'beta': [beta],
        'lambda_neg': [lambda_neg],
        'lambda_sum': [lambda_sum],
        'test_loss': [avg_test_loss],
        'test_mse': [avg_mse_test],
        'test_ce': [avg_ce_test],
        'test_neg': [avg_neg_test],
        'test_sum': [avg_sum_test],
        'acc_z1': [acc_z1],
        'acc_z2': [acc_z2]
    })
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(results_csv, index=False)

    print(f"Тест: loss={avg_test_loss:.4f}, mse={avg_mse_test:.4f}, ce={avg_ce_test:.4f}, "
          f"neg={avg_neg_test:.4f}, sum={avg_sum_test:.4f}, acc_z1={acc_z1:.4f}, acc_z2={acc_z2:.4f}")

print("\nВсе эксперименты завершены. Результаты сохранены в папке 'analysis'.")