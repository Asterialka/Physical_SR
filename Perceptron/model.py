import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import physically_plausible as pp

class SpectralNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=31):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, rgb):
        return self.fc(rgb)

class PhysicalLayer(nn.Module):
    def __init__(self, fundamental_metamer, black_basis):
        super().__init__()
        self.register_buffer('F', fundamental_metamer)
        self.register_buffer('B', black_basis)
    def forward(self, rgb, alpha):
        fundamental = torch.matmul(rgb, self.F.T)
        black = torch.matmul(alpha, self.B.T)
        return fundamental + black

class FullModel(nn.Module):
    def __init__(self, fundamental_metamer, black_basis):
        super().__init__()
        self.net = SpectralNetwork(output_dim=black_basis.shape[1])
        self.physics = PhysicalLayer(fundamental_metamer, black_basis)
    def forward(self, rgb):
        alpha = self.net(rgb)
        return self.physics(rgb, alpha)

class HyperspectralDataset(Dataset):
    def __init__(self, input_dir, sensitivity_file):
        self.input_dir = input_dir
        self.sensitivity_file = sensitivity_file
        self.file_list = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
        self.wl, self.cie_xyz = pp.load_spectral_sensitivity(sensitivity_file)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.input_dir, self.file_list[idx])
        hsi, wavelengths = pp.load_hyperspectral_data(filepath)

        if wavelengths is None:
            wavelengths = np.linspace(400, 700, hsi.shape[2])

        resampled_sensitivity = np.zeros((len(wavelengths), 3))
        for i in range(3):
            interp_func = interp1d(self.wl, self.cie_xyz[:, i], bounds_error=False, fill_value=0)
            resampled_sensitivity[:, i] = interp_func(wavelengths)

        resampled_sensitivity /= np.max(resampled_sensitivity, axis=0)

        fundamental_metamer, rgb = pp.calculate_fundamental_metamer(hsi, resampled_sensitivity)
        null_basis = pp.calculate_null_basis(hsi, resampled_sensitivity)

        rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)  
        hsi_tensor = torch.from_numpy(hsi).float().permute(2, 0, 1)  

        return rgb_tensor, hsi_tensor, fundamental_metamer, null_basis
