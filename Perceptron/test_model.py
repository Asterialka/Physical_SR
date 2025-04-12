import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import torch.nn as nn
from scipy.linalg import svd
import metrics 
import utils
from tqdm import tqdm
import physically_plausible as pp
from model import SpectralNetwork, PhysicalLayer, FullModel, HyperspectralDataset 

resources = {
    'wp_list': utils.loadmat2array('./resources/white_point_selection.mat', 'white_spectrum'),
    'name_list': open('./resources/fname_wp.txt').readlines(),
    'sensitivity' : utils.load_cie64cmf('./resources/', np.arange(400,701,9))
}

def recover(regress_matrix, regress_input, advanced_mode, resources, gt_rgb=(), exposure=1):
    
    recovery = {}
    recovery['spec'] = regress_input @ regress_matrix
    if advanced_mode['Physically_Plausible']:
        recovery['spec'] = gt_rgb*exposure @ resources['funda_mat'].T + recovery['spec'] @ resources['null_basis'].T
    
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
    return recovery

def test_model(model_path, input_dir, sensitivity_file, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Полное тестирование модели с расчетом метрик и визуализацией результатов
    
    Параметры:
        model_path (str): путь к сохраненной модели
        input_dir (str): директория с тестовыми HDF5 файлами
        sensitivity_file (str): путь к файлу с данными о чувствительности
        device (str): устройство для вычислений ('cuda' или 'cpu')
    """
    
    test_dataset = HyperspectralDataset(input_dir, sensitivity_file)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    sample_rgb, sample_hsi, sample_fm, sample_nb = test_dataset[0]
    F = torch.from_numpy(sample_fm).float().to(device)
    B = torch.from_numpy(sample_nb).float().to(device)
    
    model = FullModel(F, B).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    mrae_list = []
    rmse_list = []
    psnr_list = []
    spectral_angles = []
    de00_list = []
    
    with torch.no_grad():
        for batch_idx, (rgb, hsi, _, _) in enumerate(tqdm(test_loader, desc="Processing images")):
            rgb = rgb.to(device)
            hsi = hsi.to(device)
            
            B, C, H, W = rgb.shape
            rgb_flat = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
            
            pred_hsi_flat = model(rgb_flat)
            pred_hsi = pred_hsi_flat.reshape(H, W, 34).permute(2, 0, 1).unsqueeze(0)
            
            mrae = torch.mean(torch.abs(pred_hsi - hsi) / (torch.mean(hsi) + 1e-6))
            mrae_list.append(mrae.item())
            
            rmse = torch.sqrt(torch.mean((pred_hsi - hsi)**2))
            rmse_list.append(rmse.item())
            
            max_val = torch.max(hsi)
            mse = torch.mean((pred_hsi - hsi)**2)
            psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
            psnr_list.append(psnr.item())
            
            dot_product = torch.sum(pred_hsi * hsi, dim=1)
            pred_norm = torch.norm(pred_hsi, dim=1)
            target_norm = torch.norm(hsi, dim=1)
            sam = torch.acos(dot_product / (pred_norm * target_norm + 1e-6))
            spectral_angles.append(torch.mean(sam).item())
            
            try:
                wp_spec = utils.search_spec(test_dataset.file_list[batch_idx])
                gt_lab = metrics.spec2lab(hsi.cpu().numpy(), sensitivity_file, wp_spec)
                rec_lab = metrics.spec2lab(pred_hsi.cpu().numpy(), sensitivity_file, wp_spec)
                de00 = metrics.cal_dE00(gt_lab, rec_lab)
                de00_list.append(de00)
            except Exception as e:
                print(f"\nWarning: Could not calculate dE00 for {test_dataset.file_list[batch_idx]}: {str(e)}")
    
    print("\n=== Summary Metrics ===")
    print(f"Mean MRAE: {np.mean(mrae_list):.4f} ± {np.std(mrae_list):.4f}")
    print(f"Mean RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB ± {np.std(psnr_list):.2f}")
    print(f"Mean SAM: {np.mean(spectral_angles):.4f} rad ± {np.std(spectral_angles):.4f}")
    if de00_list:
        print(f"Mean dE00: {np.mean(de00_list):.2f} ± {np.std(de00_list):.2f}")
    
    plt.figure(figsize=(15, 4))
    
    metrics_to_plot = [
        ('MRAE', mrae_list),
        ('RMSE', rmse_list),
        ('SAM (rad)', spectral_angles)
    ]
    
    if de00_list:
        metrics_to_plot.append(('dE00', de00_list))
    
    for i, (title, values) in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), i+1)
        plt.hist(values, bins=20, alpha=0.7)
        plt.title(f'Distribution of {title}')
        plt.xlabel(title)
        if i == 0:
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mrae': np.mean(mrae_list),
        'rmse': np.mean(rmse_list),
        'psnr': np.mean(psnr_list),
        'sam': np.mean(spectral_angles),
        'de00': np.mean(de00_list) if de00_list else None,
        'std_mrae': np.std(mrae_list),
        'std_rmse': np.std(rmse_list),
        'std_psnr': np.std(psnr_list),
        'std_sam': np.std(spectral_angles),
        'std_de00': np.std(de00_list) if de00_list else None
    }

if __name__ == "__main__":
    metrics = test_model(
        model_path="./model_zoo/perceptron_50epochs.pth",
        input_dir="test_images",
        sensitivity_file="resources/ciexyz64.csv"
    )
    
    print("\nFinal Average Metrics Across All Images:")
    print(f"MRAE: {metrics['mrae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SAM: {metrics['sam']:.4f} rad")
    if metrics['de00'] is not None:
        print(f"dE00: {metrics['de00']:.2f} ± {metrics['std_de00']:.2f}")