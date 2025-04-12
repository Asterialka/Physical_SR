import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.linalg import svd
import physically_plausible as pp
from model import SpectralNetwork, PhysicalLayer, FullModel, HyperspectralDataset 

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dir = "/dataset"
    sensitivity_file = "./resources/ciexyz64.csv"
    save_path = "./model_zoo/spectral_reconstruction_model.pth"
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-3

    dataset = HyperspectralDataset(input_dir, sensitivity_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample_rgb, sample_hsi, sample_fm, sample_nb = dataset[0]
    F = torch.from_numpy(sample_fm).float().to(device)
    B = torch.from_numpy(sample_nb).float().to(device)
    model = FullModel(F, B).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_mrae = 0.0
        num_batches = 0
        for rgb_batch, hsi_batch, _, _ in dataloader:
            rgb_batch = rgb_batch.to(device)
            hsi_batch = hsi_batch.to(device)

            B, C, H, W = rgb_batch.shape
            rgb_flat = rgb_batch.permute(0, 2, 3, 1).reshape(-1, 3)  
            hsi_flat = hsi_batch.permute(0, 2, 3, 1).reshape(-1, 34) 

            optimizer.zero_grad()
            output = model(rgb_flat)
            loss = criterion(output, hsi_flat)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

if __name__ == "__main__":
    train_model()