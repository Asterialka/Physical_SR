import h5py
import pandas as pd
import numpy as np
from scipy.linalg import svd

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

def load_spectral_sensitivity(filepath):
    """Загружает данные о чувствительности"""
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] < 4:
            raise ValueError("Файл должен содержать минимум 4 столбца")
        return df.iloc[:, 0].values, np.vstack([df.iloc[:,1], df.iloc[:,2], df.iloc[:,3]]).T
    except Exception as e:
        raise ValueError(f"Ошибка загрузки {filepath}: {str(e)}")

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

def calculate_fundamental_metamer(hsi, sensitivity):
    hsi_flat = hsi.reshape(-1, hsi.shape[2])
    rgb_flat = np.dot(hsi_flat, sensitivity)
    rgb = rgb_flat.reshape(hsi.shape[0], hsi.shape[1], 3)
    mean_rgb = np.mean(rgb, axis=(0, 1))
    sts = np.dot(sensitivity.T, sensitivity)
    sts_inv = np.linalg.inv(sts)
    projection_matrix = np.dot(sensitivity, sts_inv)
    fundamental_metamer = projection_matrix * mean_rgb
    return fundamental_metamer, rgb

def calculate_null_basis(hsi, sensitivity):
    spectral_data = hsi.reshape(34, -1).T
    P = sensitivity @ np.linalg.inv(sensitivity.T @ sensitivity) @ sensitivity.T
    null_basis = np.eye(34) - P
    U, s, Vh = svd(null_basis, full_matrices=False)
    null_basis = Vh[3:].T
    return null_basis

