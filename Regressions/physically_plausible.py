import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy.linalg import svd
from scipy.linalg import orth

def debug_h5_structure(filepath):
    """Выводит структуру HDF5 файла для диагностики"""
    print(f"\nАнализ структуры файла: {filepath}")
    with h5py.File(filepath, 'r') as f:
        print("Доступные datasets/groups:")
        def print_attrs(name, obj):
            print(f"{name}: shape={obj.shape if hasattr(obj, 'shape') else 'group'}, dtype={obj.dtype if hasattr(obj, 'dtype') else 'group'}")
            for k, v in obj.attrs.items():
                print(f"   attr: {k} = {v}")
        f.visititems(print_attrs)

def find_hsi_dataset(h5file):
    """Находит dataset с гиперспектральными данными"""
    candidates = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
            candidates.append((name, obj.shape))
    h5file.visititems(visitor)
    
    if not candidates:
        raise ValueError("Не найдено подходящих 3D datasets")
    best_match = max(candidates, key=lambda x: x[1][0])  
    return best_match[0]

def load_hyperspectral_data(filepath):
    """Загружает данные из HDF5 файла"""
    try:
        with h5py.File(filepath, 'r') as f:
            dataset_name = find_hsi_dataset(f)
            print(f"Используется dataset: {dataset_name}")
            data = f[dataset_name][:] 
            data = np.transpose(data, (1, 2, 0))
            wavelengths = None
            for name in ['wavelengths', 'bands', 'lambda', 'wave', 'wvl']:
                if name in f:
                    wavelengths = f[name][:]
                    break
            
            return data, wavelengths
    except Exception as e:
        debug_h5_structure(filepath)
        raise ValueError(f"Ошибка загрузки {filepath}: {str(e)}")
    
def calculate_null_basis(hsi, S):
    n, k = S.shape
    if k != 3:
        raise ValueError("Не RGB")
    PS = S @ np.linalg.inv(S.T @ S) @ S.T
    I = np.eye(n)
    PB = I - PS
    B = orth(PB)
    print("Проверка ортогональности B.T @ S = ", B.T @ S)
    if B.shape[1] != n - 3:
        print(f"Предупреждение: Ожидалось {n - 3} базисных векторов, получено {B.shape[1]}.")
    return B 

def load_cie64cmf(directory, target_wavelength=np.arange(400,701, 9)):
    path_name = os.path.join(directory, 'ciexyz64.csv')
    cmf = np.array(pd.read_csv(path_name))[:,1:]
    lambda_cmf = np.array(pd.read_csv(path_name))[:,0] 
    
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

def calculate_fundamental_metamer(hsi, sensitivity):
    hsi_flat = hsi.reshape(-1, hsi.shape[2])  
    rgb_flat = np.dot(hsi_flat, sensitivity)  
    rgb = rgb_flat.reshape(hsi.shape[0], hsi.shape[1], 3) 

    pinv = np.linalg.pinv(sensitivity.T @ sensitivity)
    funda_mat = sensitivity @ pinv
    return funda_mat, rgb

def main_processing():
    input_dir = "test_images"
    output_dir = "resources"
    sensitivity_file = "resources/ciexyz64.csv"
    sensitivity_files = "resources/"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        h5_files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
        if not h5_files:
            raise ValueError(f"Не найдено .h5 файлов в {input_dir}")
        
        for filename in tqdm(h5_files, desc="Обработка файлов"):
            filepath = os.path.join(input_dir, filename)
            try:
                print(f"\nНачало обработки: {filename}")
                hsi, wavelengths = load_hyperspectral_data(filepath)
                print(f"Загружены данные: форма {hsi.shape}, каналов: {hsi.shape[2]}")
                if wavelengths is None:
                    wavelengths = np.linspace(400, 700, hsi.shape[2])  
                    print(f"Используются длины волн по умолчанию: {wavelengths[0]}..{wavelengths[-1]} нм")         
                resampled_sensitivity = load_cie64cmf(sensitivity_files)
                fundamental_metamer, rgb = calculate_fundamental_metamer(
                    hsi, resampled_sensitivity)
                null_basis = calculate_null_basis(hsi, resampled_sensitivity)
                basename = os.path.splitext(filename)[0]
                np.save(os.path.join(output_dir, f"{basename}_fundamental_metamer.npy"), fundamental_metamer)
                np.save(os.path.join(output_dir, f"{basename}_rgb.npy"), rgb)
                np.save(os.path.join(output_dir, f"{basename}_null_basis"), null_basis)
                plt.imshow(rgb / np.max(rgb)) 
                plt.title(f"RGB: {filename}")
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"{basename}_rgb.png"), bbox_inches='tight', dpi=150)
                plt.close()
            
                print(f"Успешно обработан: {filename}")
                
            except Exception as e:
                print(f"\nОшибка при обработке {filename}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
        return

if __name__ == "__main__":
    main_processing()
    print("\nОбработка завершена")