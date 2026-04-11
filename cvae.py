"""
Модель нейросети CVAE
"""


import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    def __init__(self, geom_dim, hydro_dim, latent_dim=128, hidden_dim=128, num_classes=5):
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
        self.cont_head = nn.Linear(hidden_dim, 6)          # A, r1, r, r0, h, L
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

def loss_function(cont_pred, logits_z1, logits_z2,
                  cont_target, z1_target, z2_target,
                  mean, logvar,
                  cont_mean, cont_scale, weight_z1=None, weight_z2=None,
                  beta=1.0, lambda_neg=0, lambda_sum=1, lambda_ce=0):
    mse = nn.functional.mse_loss(cont_pred, cont_target, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # взвешенная кросс энтропия
    ce_z1 = nn.functional.cross_entropy(logits_z1, z1_target, weight=weight_z1, reduction='sum')
    ce_z2 = nn.functional.cross_entropy(logits_z2, z2_target, weight=weight_z2, reduction='sum')

    # Денормализация предсказаний и целевых значений (для штрафов)
    cont_pred_denorm = cont_pred * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)
    cont_target_denorm = cont_target * cont_scale.unsqueeze(0) + cont_mean.unsqueeze(0)

    A_pred = cont_pred_denorm[:, 0]
    r1_pred = torch.expm1(cont_pred_denorm[:, 1])
    r2_pred = A_pred - r1_pred
    r_pred = torch.expm1(cont_pred_denorm[:, 2])
    r0_pred = torch.expm1(cont_pred_denorm[:, 3])
    h_pred = torch.expm1(cont_pred_denorm[:, 4])
    L_pred = torch.expm1(cont_pred_denorm[:, 5])

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

    total_loss = 10 * mse + beta * kl + lambda_ce * (ce_z1 + ce_z2) + lambda_neg * neg_penalty + lambda_sum * sum_penalty
    return total_loss, mse, ce_z1 + ce_z2, kl, neg_penalty, sum_penalty
