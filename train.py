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

df = pd.read_csv('params.csv', index_col=0)
df['z1'] = df['z1'].astype(int)
df['z2'] = df['z2'].astype(int)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')

# Признаки, которые будем логарифмировать (сильно скошенные)
skewed_features = ['r1', 'r', 'r0', 'h', 'L']
for feat in skewed_features:
    df[feat + '_log'] = np.log1p(df[feat])  # log(1+x) для положительных x

geom_cols = ['A'] + [f+'_log' for f in skewed_features] + ['z1', 'z2']
hydro_cols = ['efficiency', 'mass_flow']
data = df[geom_cols].values
n_sample = 1000

# некоторые дополнительные признаки для упрощения зависимости
df['r_sum'] = df['r1'] + df['r2']
df['r_ratio'] = df['r1'] / df['r2']
df['h_rel'] = df['h'] / df['A']
df['L_rel'] = df['L'] / df['A']
df['windings'] = df['L'] / df['h']
df['modul'] = 2 * df['A'] / (df['z1'] + df['z2'])

# Линейная комбинация для КПД (пример)
efficiency = (0.1 * df['r_ratio'] +
              0.05 * df['h_rel'] * 10 +
              0.02 * df['windings'] +
              0.1 * df['modul'] / 50 +
              0.2 * (df['z1'] / 8) +
              0.2 * (df['z2'] / 8) +
              0.3 * (df['A'] / 600) +
              np.random.normal(0, 0.02, len(df)))  # шум

efficiency = np.clip(efficiency, 0.5, 1.0)

# Линейная комбинация для массового расхода (положительный)
mass_flow = (0.5 * df['A'] / 100 +
             0.3 * df['h'] / 1000 +
             0.1 * df['L'] / 1000 +
             0.1 * df['modul'] +
             np.random.normal(0, 2, len(df)))
mass_flow = np.abs(mass_flow) + 1

n_samples = len(df)
np.random.seed(42)
# efficiency = np.random.uniform(0.5, 1.0, n_samples)   # КПД
# mass_flow = np.random.uniform(1, 100, n_samples)      # массовый расход
df['efficiency'] = efficiency
df['mass_flow'] = mass_flow
hydro_data = np.column_stack([efficiency, mass_flow])

scaler_geom = StandardScaler()
scaler_hydro = StandardScaler()

X = df[geom_cols].values      # геометрия
y = df[hydro_cols].values     # гидродинамика

# Разделение: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train_scaled = scaler_geom.fit_transform(X_train)
X_val_scaled = scaler_geom.transform(X_val)
X_test_scaled = scaler_geom.transform(X_test)

y_train_scaled = scaler_hydro.fit_transform(y_train)
y_val_scaled = scaler_hydro.transform(y_val)
y_test_scaled = scaler_hydro.transform(y_test)

# Сохраняем средние и масштабы для непрерывных признаков (первые 7)
cont_mean = torch.FloatTensor(scaler_geom.mean_[:6])
cont_scale = torch.FloatTensor(scaler_geom.scale_[:6])

# Целевые для классификации z1, z2 (индексы 7,8 в X)
z1_train = torch.LongTensor(X_train[:, 6].astype(int) - 4)
z2_train = torch.LongTensor(X_train[:, 7].astype(int) - 4)
z1_val = torch.LongTensor(X_val[:, 6].astype(int) - 4)
z2_val = torch.LongTensor(X_val[:, 7].astype(int) - 4)
z1_test = torch.LongTensor(X_test[:, 6].astype(int) - 4)
z2_test = torch.LongTensor(X_test[:, 7].astype(int) - 4)

# Создаём тензоры
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train_scaled)
X_val_t = torch.FloatTensor(X_val_scaled)
y_val_t = torch.FloatTensor(y_val_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)
y_test_t = torch.FloatTensor(y_test_scaled)

# Датасеты и загрузчики
train_dataset = TensorDataset(X_train_t, y_train_t, z1_train, z2_train)
val_dataset = TensorDataset(X_val_t, y_val_t, z1_val, z2_val)
test_dataset = TensorDataset(X_test_t, y_test_t, z1_test, z2_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# classes_z1 = np.unique(z1_train.numpy())
# weights_z1 = compute_class_weight('balanced', classes=classes_z1, y=z1_train.numpy())
# class_weights_z1 = torch.FloatTensor(weights_z1)
#
# classes_z2 = np.unique(z2_train.numpy())
# weights_z2 = compute_class_weight('balanced', classes=classes_z2, y=z2_train.numpy())
# class_weights_z2 = torch.FloatTensor(weights_z2)

geom_dim = 8
hydro_dim = 2
latent_dim = 7
model = ConditionalVAE(geom_dim, hydro_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
train_losses = []
val_losses = []
mse_losses = []
ce_losses = []
kl_losses = []
neg_penalties = []
sum_penalties = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_mse = 0
    total_kl = 0
    total_ce = 0
    total_neg = 0
    total_sum = 0
    for batch in train_loader:
        # geom, hydro, _, _ = next(iter(train_loader))
        # mean, logvar = model.encode(geom, hydro)
        # print("Mean stats: mean={:.4f}, std={:.4f}".format(mean.mean().item(), mean.std().item()))
        # print("Var stats: mean={:.4f}, std={:.4f}".format(logvar.exp().mean().item(), logvar.exp().std().item()))

        geom, hydro, z1t, z2t = batch
        optimizer.zero_grad()
        cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
        cont_target = geom[:, :6]  # первые 6 столбцов (A, r1, r, r0, h, L)
        loss, mse, ce, kl, neg, sum_pen = loss_function(
            cont_pred, logits_z1, logits_z2,
            cont_target, z1t, z2t,
            mean, logvar,
            cont_mean, cont_scale,
            # weight_z1=class_weights_z1,
            # weight_z2=class_weights_z2
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

    # Валидация
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            geom, hydro, z1t, z2t = batch
            cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
            cont_target = geom[:, :6]
            loss, _, _, _, _, _ = loss_function(
                cont_pred, logits_z1, logits_z2,
                cont_target, z1t, z2t,
                mean, logvar,
                cont_mean, cont_scale,
                # weight_z1=class_weights_z1,
                # weight_z2=class_weights_z2
            )
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataset)
    val_losses.append(avg_val_loss)


plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(train_losses)
plt.title('Total Loss')

plt.subplot(2, 3, 2)
plt.plot(mse_losses)
plt.title('MSE (continuous)')

plt.subplot(2, 3, 3)
plt.plot(ce_losses)
plt.title('Cross-Entropy (z1+z2)')

plt.subplot(2, 3, 4)
plt.plot(kl_losses)
plt.title('KL Divergence')

plt.subplot(2, 3, 5)
plt.plot(neg_penalties)
plt.title('Negative Penalty')

plt.subplot(2, 3, 6)
plt.plot(sum_penalties)
plt.title('Sum Penalty')

plt.tight_layout()
plt.show()

model.eval()
total_test_loss = 0
mse_total = 0
ce_total = 0
neg_total = 0
sum_total = 0
examples_shown = False
all_cont_pred = []
all_cont_true = []
all_z1_pred = []
all_z1_true = []
all_z2_pred = []
all_z2_true = []
with torch.no_grad():
    for batch in test_loader:
        geom, hydro, z1t, z2t = batch
        cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
        cont_target = geom[:, :6]
        loss, mse, ce, kl, neg, sum_pen = loss_function(
            cont_pred, logits_z1, logits_z2,
            cont_target, z1t, z2t,
            mean, logvar,
            cont_mean, cont_scale,
            # weight_z1=class_weights_z1,
            # weight_z2=class_weights_z2
        )
        total_test_loss += loss.item()
        mse_total += mse.item()
        ce_total += ce.item()
        neg_total += neg.item()
        sum_total += sum_pen.item()

        # Денормализация
        cont_pred_denorm = cont_pred * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)
        cont_target_denorm = cont_target * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)

        all_cont_pred.append(cont_pred_denorm.cpu().numpy())
        all_cont_true.append(cont_target_denorm.cpu().numpy())

        pred_z1 = torch.argmax(logits_z1, dim=1).cpu().numpy() + 4
        pred_z2 = torch.argmax(logits_z2, dim=1).cpu().numpy() + 4
        true_z1 = z1t.cpu().numpy() + 4
        true_z2 = z2t.cpu().numpy() + 4

        all_z1_pred.extend(pred_z1)
        all_z1_true.extend(true_z1)
        all_z2_pred.extend(pred_z2)
        all_z2_true.extend(true_z2)

        # Выводим примеры только для первого батча
        if not examples_shown:
            # Выведем первые 5 примеров из батча
            n_show = min(5, len(cont_pred))
            print("\nПримеры предсказаний на тестовой выборке:")
            for i in range(n_show):
                print(f"\nПример {i + 1}:")
                print(f"  Предсказано: A={cont_pred_denorm[i, 0].item():.2f}, "
                    f"r1={torch.expm1(cont_pred_denorm[i, 1]).item():.2f}, "
                    f"r2={cont_pred_denorm[i, 0].item() - torch.expm1(cont_pred_denorm[i, 1]).item():.2f}, "
                    f"r={torch.expm1(cont_pred_denorm[i, 2]).item():.2f}, "
                    f"r0={torch.expm1(cont_pred_denorm[i, 3]).item():.2f}, "
                    f"h={torch.expm1(cont_pred_denorm[i, 4]).item():.2f}, "
                    f"L={torch.expm1(cont_pred_denorm[i, 5]).item():.2f}, "
                    f"z1={pred_z1[i].item()}, z2={pred_z2[i].item()}")
                print(
                    f"  Истинно:    A={cont_target_denorm[i, 0].item():.2f}, "
                    f"r1={torch.expm1(cont_target_denorm[i, 1]).item():.2f}, "
                    f"r2={cont_target_denorm[i, 0].item() - torch.expm1(cont_target_denorm[i, 1]).item():.2f}, "
                    f"r={torch.expm1(cont_target_denorm[i, 2]).item():.2f}, "
                    f"r0={torch.expm1(cont_target_denorm[i, 3]).item():.2f}," 
                    f"h={torch.expm1(cont_target_denorm[i, 4]).item():.2f}, "
                    f"L={torch.expm1(cont_target_denorm[i, 5]).item():.2f}, " 
                    f"z1={true_z1[i].item()}, z2={true_z2[i].item()}")

            examples_shown = True

avg_test_loss = total_test_loss / len(test_dataset)
avg_mse = mse_total / len(test_dataset)
avg_ce = ce_total / len(test_dataset)
avg_neg = neg_total / len(test_dataset)
avg_sum = sum_total / len(test_dataset)

# Объединяем все батчи
all_cont_pred = np.concatenate(all_cont_pred, axis=0)
all_cont_true = np.concatenate(all_cont_true, axis=0)

# Преобразуем логарифмированные признаки (индексы 1-6) обратно
for idx in range(1,6):
    all_cont_pred[:, idx] = np.expm1(all_cont_pred[:, idx])
    all_cont_true[:, idx] = np.expm1(all_cont_true[:, idx])

# Вычисляем ошибки по каждому параметру
param_names = ['A', 'r1', 'r', 'r0', 'h', 'L']
mae_per_param = np.mean(np.abs(all_cont_pred - all_cont_true), axis=0)
rmse_per_param = np.sqrt(np.mean((all_cont_pred - all_cont_true)**2, axis=0))
mape_per_param = np.mean(np.abs((all_cont_true - all_cont_pred) / (all_cont_true)), axis=0) * 100

print("\nОшибки по непрерывным параметрам на тестовой выборке:")
for i, name in enumerate(param_names):
    print(f"{name}: MAE = {mae_per_param[i]:.3f}, RMSE = {rmse_per_param[i]:.3f}, {name}: MAPE = {mape_per_param[i]:.2f}%")

# Построение scatter plots
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()
for i in range(6):
    ax = axes[i]
    ax.scatter(all_cont_true[:, i], all_cont_pred[:, i], alpha=0.3, s=10)
    min_val = min(all_cont_true[:, i].min(), all_cont_pred[:, i].min())
    max_val = max(all_cont_true[:, i].max(), all_cont_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    ax.set_xlabel(f'True {param_names[i]}')
    ax.set_ylabel(f'Predicted {param_names[i]}')
    ax.set_title(f'{param_names[i]} (MAE={mae_per_param[i]:.2f})')
    ax.grid(True, alpha=0.3)
for j in range(6, 9):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

print(f"\nТестовые результаты:")
print(f"Total Loss: {avg_test_loss:.4f}")
print(f"MSE (continuous): {avg_mse:.4f}")
print(f"Cross-Entropy (z1+z2): {avg_ce:.4f}")
print(f"Negative Penalty: {avg_neg:.4f}")
print(f"Sum Penalty: {avg_sum:.4f}")

# Дополнительно: точность классификации z1, z2
correct_z1 = 0
correct_z2 = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        geom, hydro, z1t, z2t = batch
        _, logits_z1, logits_z2, _, _ = model(geom, hydro)
        pred_z1 = torch.argmax(logits_z1, dim=1)
        pred_z2 = torch.argmax(logits_z2, dim=1)
        correct_z1 += (pred_z1 == z1t).sum().item()
        correct_z2 += (pred_z2 == z2t).sum().item()
        total += z1t.size(0)

print(f"Accuracy z1: {correct_z1/total:.4f}")
print(f"Accuracy z2: {correct_z2/total:.4f}")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), 'cvae_geom.pth')
with open('scaler_geom.pkl', 'wb') as f:
    pickle.dump(scaler_geom, f)
with open('scaler_hydro.pkl', 'wb') as f:
    pickle.dump(scaler_hydro, f)
