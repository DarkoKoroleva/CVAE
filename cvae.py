import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('params.csv', index_col=0)  # укажите правильный путь
# z1 и z2 должны быть целыми, но могут быть float из-за чтения; приведём к int
df['z1'] = df['z1'].astype(int)
df['z2'] = df['z2'].astype(int)
df.drop(['phi01', 'phi02'], axis=1, inplace=True, errors='ignore')

# Массив геометрических признаков в порядке: A, r1, r2, r, r0, h, L, z1, z2
geom_cols = ['A', 'r1', 'r2', 'r', 'r0', 'h', 'L', 'z1', 'z2']
data = df[geom_cols].values

n_samples = len(df)
np.random.seed(42)
efficiency = np.random.uniform(0.5, 1.0, n_samples)   # КПД
mass_flow = np.random.uniform(1, 100, n_samples)      # массовый расход
hydro_data = np.column_stack([efficiency, mass_flow])

scaler_geom = StandardScaler()
scaler_hydro = StandardScaler()

data_scaled = scaler_geom.fit_transform(data)          # все 9 признаков масштабированы
hydro_scaled = scaler_hydro.fit_transform(hydro_data)

# Сохраняем средние и масштабы для непрерывных признаков (первые 7)
cont_mean = torch.FloatTensor(scaler_geom.mean_[:7])
cont_scale = torch.FloatTensor(scaler_geom.scale_[:7])

# Целевые для классификации z1, z2 (сдвиг к 0..4)
z1_target = torch.LongTensor(df['z1'].values - 4)
z2_target = torch.LongTensor(df['z2'].values - 4)

# Тензоры
geom_tensor = torch.FloatTensor(data_scaled)
hydro_tensor = torch.FloatTensor(hydro_scaled)

dataset = TensorDataset(geom_tensor, hydro_tensor, z1_target, z2_target)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class ConditionalVAE(nn.Module):
    def __init__(self, geom_dim, hydro_dim, latent_dim=8, hidden_dim=64, num_classes=5):
        super(ConditionalVAE, self).__init__()
        self.geom_dim = geom_dim
        self.hydro_dim = hydro_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        encoder_input_dim = geom_dim + hydro_dim
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        decoder_input_dim = latent_dim + hydro_dim
        self.decoder_shared = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Головы
        self.cont_head = nn.Linear(hidden_dim, 7)          # A, r1, r2, r, r0, h, L
        self.z1_head = nn.Linear(hidden_dim, num_classes)
        self.z2_head = nn.Linear(hidden_dim, num_classes)

    def encode(self, geom, hydro):
        x = torch.cat([geom, hydro], dim=1)
        h = self.encoder(x)
        return self.mean_layer(h), self.logvar_layer(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z, hydro):
        x = torch.cat([z, hydro], dim=1)
        h = self.decoder_shared(x)
        cont = self.cont_head(h)
        logits_z1 = self.z1_head(h)
        logits_z2 = self.z2_head(h)
        return cont, logits_z1, logits_z2

    def forward(self, geom, hydro):
        mean, logvar = self.encode(geom, hydro)
        z = self.reparameterize(mean, logvar)
        cont, logits_z1, logits_z2 = self.decode(z, hydro)
        return cont, logits_z1, logits_z2, mean, logvar


geom_dim = 9
hydro_dim = 2
latent_dim = 8
model = ConditionalVAE(geom_dim, hydro_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(cont_pred, logits_z1, logits_z2,
                  cont_target, z1_target, z2_target,
                  mean, logvar,
                  cont_mean, cont_scale,
                  beta=1.0, lambda_neg=1.0, lambda_sum=10.0):
    mse = nn.functional.mse_loss(cont_pred, cont_target, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # Кросс-энтропия для z1 и z2
    ce_z1 = nn.functional.cross_entropy(logits_z1, z1_target, reduction='sum')
    ce_z2 = nn.functional.cross_entropy(logits_z2, z2_target, reduction='sum')

    # Денормализация предсказаний и целевых значений (для штрафов)
    cont_pred_denorm = cont_pred * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)

    A_pred = cont_pred_denorm[:, 0]
    r1_pred = cont_pred_denorm[:, 1]
    r2_pred = cont_pred_denorm[:, 2]
    r_pred = cont_pred_denorm[:, 3]
    r0_pred = cont_pred_denorm[:, 4]
    h_pred = cont_pred_denorm[:, 5]
    L_pred = cont_pred_denorm[:, 6]

    # Штраф за отрицательные значения
    neg_penalty = (torch.relu(-A_pred) ** 2).sum() + \
                  (torch.relu(-r1_pred) ** 2).sum() + \
                  (torch.relu(-r2_pred) ** 2).sum() + \
                  (torch.relu(-r_pred) ** 2).sum() + \
                  (torch.relu(-r0_pred) ** 2).sum() + \
                  (torch.relu(-h_pred) ** 2).sum() + \
                  (torch.relu(-L_pred) ** 2).sum()

    # Штраф за нарушение r1 + r2 = A
    sum_penalty = torch.mean((r1_pred + r2_pred - A_pred)**2)

    total_loss = mse + ce_z1 + ce_z2 + beta * kl + lambda_neg * neg_penalty + lambda_sum * sum_penalty
    return total_loss, mse, ce_z1 + ce_z2, kl, neg_penalty, sum_penalty


epochs = 200
train_losses = []
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
    for batch in dataloader:
        geom, hydro, z1t, z2t = batch
        optimizer.zero_grad()
        cont_pred, logits_z1, logits_z2, mean, logvar = model(geom, hydro)
        cont_target = geom[:, :7]  # первые 7 столбцов (A, r1, r2, r, r0, h, L)
        loss, mse, ce, kl, neg, sum_pen = loss_function(
            cont_pred, logits_z1, logits_z2,
            cont_target, z1t, z2t,
            mean, logvar,
            cont_mean, cont_scale,
            beta=0.1, lambda_neg=1.0, lambda_sum=10.0
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_ce += ce.item()
        total_kl += kl.item()
        total_neg += neg.item()
        total_sum += sum_pen.item()

    avg_loss = total_loss / len(dataset)
    avg_mse = total_mse / len(dataset)
    avg_ce = total_ce / len(dataset)
    avg_kl = total_kl / len(dataset)
    avg_neg = total_neg / len(dataset)
    avg_sum = total_sum / len(dataset)

    train_losses.append(avg_loss)
    mse_losses.append(avg_mse)
    ce_losses.append(avg_ce)
    kl_losses.append(avg_kl)
    neg_penalties.append(avg_neg)
    sum_penalties.append(avg_sum)

    # print(
    #     f'Epoch {epoch + 1}: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}, KL={avg_kl:.4f}, Neg={avg_neg:.4f}, Sum={avg_sum:.4f}')


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
with torch.no_grad():
    # желаемые гидродинамические характеристики (пример)
    desired_hydro = np.array([[0.85, 10.0]])
    desired_hydro_scaled = scaler_hydro.transform(desired_hydro)
    hydro_t = torch.FloatTensor(desired_hydro_scaled)

    n_gen = 5
    z = torch.randn(n_gen, latent_dim)
    hydro_repeated = hydro_t.repeat(n_gen, 1)
    cont_pred, logits_z1, logits_z2 = model.decode(z, hydro_repeated)

    # Денормализуем непрерывные
    cont_pred_denorm = cont_pred * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)
    A_gen = cont_pred_denorm[:, 0].numpy()
    r1_gen = cont_pred_denorm[:, 1].numpy()
    r2_gen = cont_pred_denorm[:, 2].numpy()
    r_gen = cont_pred_denorm[:, 3].numpy()
    r0_gen = cont_pred_denorm[:, 4].numpy()
    h_gen = cont_pred_denorm[:, 5].numpy()
    L_gen = cont_pred_denorm[:, 6].numpy()

    # Классы для z1, z2
    z1_probs = torch.softmax(logits_z1, dim=1)
    z2_probs = torch.softmax(logits_z2, dim=1)
    z1_gen = torch.argmax(z1_probs, dim=1).numpy() + 4
    z2_gen = torch.argmax(z2_probs, dim=1).numpy() + 4

    print("\nСгенерированные геометрические параметры:")
    for i in range(n_gen):
        print(f"Вариант {i + 1}: A={A_gen[i]:.2f}, r1={r1_gen[i]:.2f}, r2={r2_gen[i]:.2f}, "
              f"r={r_gen[i]:.2f}, r0={r0_gen[i]:.2f}, h={h_gen[i]:.2f}, L={L_gen[i]:.2f}, "
              f"z1={z1_gen[i]}, z2={z2_gen[i]}")


# torch.save(model.state_dict(), 'cvae_geom_physical.pth')
# with open('scaler_geom.pkl', 'wb') as f:
#     pickle.dump(scaler_geom, f)
# with open('scaler_hydro.pkl', 'wb') as f:
#     pickle.dump(scaler_hydro, f)