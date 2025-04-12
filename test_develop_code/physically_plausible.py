import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from scipy.interpolate import interp1d
import torch.nn as nn
import h5py
from scipy.linalg import orth
import torch
import numpy as np
from copy import copy
from pandas import read_csv

def calculate_null_basis(S):
    n, k = S.shape
    if k != 3:
        raise ValueError("Не RGB")
    PS = S @ np.linalg.inv(S.T @ S) @ S.T
    I = np.eye(n)
    PB = I - PS
    B = orth(PB)
    if B.shape[1] != n - 3:
        print(f"Предупреждение: Ожидалось {n - 3} базисных векторов, получено {B.shape[1]}.")
    return B 

def calculate_fundamental_metamer(sensitivity):
    pinv = np.linalg.pinv(sensitivity.T @ sensitivity)
    funda_mat = sensitivity @ pinv
    return funda_mat

def load_cie64cmf(directory, target_wavelength=np.arange(400,701, 9)):
    path_name = os.path.join(directory, 'ciexyz64.csv')
    cmf = np.array(read_csv(path_name))[:,1:]
    lambda_cmf = np.array(read_csv(path_name))[:,0] 
    
    cmf = interpolate(cmf, lambda_cmf, target_wavelength)
    cmf = cmf / np.max(np.sum(cmf, 0))
    return cmf

def interpolate(data, data_waveL, targeted_waveL):
    
    assert data.shape[0] == data_waveL.size, 'Wavelength sequence mismatch with data'
    
    targeted_bounds = [np.min(targeted_waveL), np.max(targeted_waveL)]
    data_bounds = [np.min(data_waveL), np.max(data_waveL)]
    
    assert data_bounds[0] <= targeted_bounds[0], 'targeted wavelength range must be within the original wavelength range'
    assert data_bounds[1] >= targeted_bounds[1], 'targeted wavelength range must be within the original wavelength range'
    
    dim_new_data = list(data.shape)
    dim_new_data[0] = len(targeted_waveL)
    new_data = np.empty(dim_new_data)
    for i in range(len(targeted_waveL)):

        relative_L = data_waveL - targeted_waveL[i]
        
        if 0 in relative_L:
            floor = np.argmax( relative_L == 0 )
            new_data[i,...] = data[floor,...]
        
        else:
            floor = np.argmax( relative_L >= 0 ) -1
            interval = data_waveL[floor+1] - data_waveL[floor]
            portion = (targeted_waveL[i] - data_waveL[floor])/interval
            new_data[i,...] = portion*data[floor,...] + (1-portion)*data[floor+1,...]
    
    return new_data 


def calculate_rgb(hsi, sensitivity):
    hsi_flat = hsi.reshape(-1, hsi.shape[2])  # (height*width, n_channels)
    rgb_flat = np.dot(hsi_flat, sensitivity)  # (height*width, 3)
    rgb = rgb_flat.reshape(hsi.shape[0], hsi.shape[1], 3)  # (height, width, 3)
    return rgb
