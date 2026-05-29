import torch
import torch.nn as nn


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class LinearSin(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = SinActivation()

    def forward(self, x):
        return self.activation(self.linear(x))


class ExecutablePINN(nn.Module):
    """Paper-specific executable approximation of Wang 2024 PINN.

    F maps 16 engineered charge-curve features plus cycle index to SOH.
    G maps constructed degradation-state variables to a degradation dynamics rate.
    """

    def __init__(self, input_dim=17, cycle_feature_index=16):
        super().__init__()
        self.input_dim = input_dim
        self.cycle_feature_index = cycle_feature_index
        self.F = nn.Sequential(
            LinearSin(input_dim, 60),
            LinearSin(60, 60),
            nn.Linear(60, 32),
            LinearSin(32, 32),
            nn.Linear(32, 1),
        )
        self.G = nn.Sequential(
            LinearSin(4, 60),
            LinearSin(60, 60),
            nn.Linear(60, 1),
        )

    def construct_g_input(self, x, predicted_soh):
        scaled_cycle = x[:, self.cycle_feature_index:self.cycle_feature_index + 1]
        other = torch.cat([x[:, :self.cycle_feature_index], x[:, self.cycle_feature_index + 1:]], dim=1)
        first_order_feature_summary = other.mean(dim=1, keepdim=True)
        capacity_loss_proxy = 1.0 - predicted_soh
        return torch.cat([predicted_soh, scaled_cycle, first_order_feature_summary, capacity_loss_proxy], dim=1)

    def forward(self, x):
        raw = self.F(x)
        predicted_soh = 1.2 * torch.sigmoid(raw)
        g_input = self.construct_g_input(x, predicted_soh)
        dynamics_rate = self.G(g_input)
        return predicted_soh, dynamics_rate

    def predict_soh(self, x):
        return self.forward(x)[0]
