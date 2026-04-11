"""
Точка входа: задаем желаемую гидродинамику -> получаем геометрию
"""


import torch
import pickle
import numpy as np
from cvae import ConditionalVAE


model = ConditionalVAE(geom_dim=9, hydro_dim=2, latent_dim=8)
model.load_state_dict(torch.load('train/cvae_geom.pth'))
model.eval()

with open('train/scaler_geom.pkl', 'rb') as f:
    scaler_geom = pickle.load(f)
with open('train/scaler_hydro.pkl', 'rb') as f:
    scaler_hydro = pickle.load(f)

# 3. Извлекаем средние и масштабы для непрерывных признаков (первые 7)
cont_mean = torch.FloatTensor(scaler_geom.mean_[:7])
cont_scale = torch.FloatTensor(scaler_geom.scale_[:7])

# 4. Желаемая гидродинамика
desired_hydro = np.array([[0.85, 30.0]])  # КПД, массовый расход
hydro_scaled = scaler_hydro.transform(desired_hydro)
hydro_t = torch.FloatTensor(hydro_scaled)

# 5. Генерация нескольких вариантов
n_gen = 5
z = torch.randn(n_gen, model.latent_dim)  # latent_dim должно быть определено в модели
hydro_repeated = hydro_t.repeat(n_gen, 1)

with torch.no_grad():
    cont_pred, logits_z1, logits_z2 = model.decode(z, hydro_repeated)

# 6. Денормализация непрерывных признаков
cont_original = cont_pred * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)
A = cont_original[:, 0].numpy()
r1 = cont_original[:, 1].numpy()
r2 = cont_original[:, 2].numpy()
r = cont_original[:, 3].numpy()
r0 = cont_original[:, 4].numpy()
h = cont_original[:, 5].numpy()
L = cont_original[:, 6].numpy()

# 7. Получение целых z1, z2
z1_probs = torch.softmax(logits_z1, dim=1)
z2_probs = torch.softmax(logits_z2, dim=1)
z1_pred = torch.argmax(z1_probs, dim=1).numpy() + 4
z2_pred = torch.argmax(z2_probs, dim=1).numpy() + 4

# 8. Вывод результатов
for i in range(n_gen):
    print(f"Вариант {i + 1}: A={A[i]:.2f}, r1={r1[i]:.2f}, r2={r2[i]:.2f}, "
          f"r={r[i]:.2f}, r0={r0[i]:.2f}, h={h[i]:.2f}, L={L[i]:.2f}, "
          f"z1={z1_pred[i]}, z2={z2_pred[i]}")
