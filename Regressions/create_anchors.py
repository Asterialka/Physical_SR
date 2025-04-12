import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.io import savemat
import os
from tqdm import tqdm

def load_h5_data(file_path):
    """Загрузка гиперспектральных данных из HDF5 файла"""
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
        # Уточните структуру вашего файла - укажите правильные ключи
        # Пример для типичной структуры: 
        # {'cube': (height, width, bands), 'wavelengths': (bands,)}
            data = np.array(f[key][:])  # Получаем гиперкуб
            wavelengths = np.array(f['wavelengths'][:]) if 'wavelengths' in f else None
    return data, wavelengths

def generate_anchors_from_h5_dataset(h5_files_list, num_anchors=1024):
    """Генерация якорей из набора HDF5 файлов"""
    all_spectra = []
    
    for file_path in h5_files_list:
        # Загрузка данных из HDF5
        print("file_path = ", file_path)
        hyperspectral_data, _ = load_h5_data(file_path)
        bands, height, width = hyperspectral_data.shape
        # Преобразование в 2D массив [pixels, bands]
        spectra = hyperspectral_data.reshape(-1, bands)
        print("spectra shape = ",  spectra.shape)
        all_spectra.append(spectra)
        print("all_spectr = ", len(all_spectra))
    
    # Объединение всех спектров
    all_spectra = np.vstack(all_spectra)
    
    # Удаление возможных NaN значений
    all_spectra = all_spectra[~np.isnan(all_spectra).any(axis=1)]
    
    # Кластеризация
    kmeans = MiniBatchKMeans(n_clusters=num_anchors, 
                            batch_size=10, 
                            random_state=42,
                            n_init=3)
    print("Начинаю обучение..")
    #kmeans.fit(all_spectra)
    n_iter = 10
    for _ in tqdm(range(n_iter), desc="Training MiniBatchKMeans"):
        kmeans.partial_fit(all_spectra)
        print(f"Inertia: {kmeans.inertia_:.2f}", end="\r")
    return kmeans.cluster_centers_

def prepare_custom_anchors(h5_directory, output_mat_path):
    """Подготовка кастомных якорей для HDF5 датасета"""
    # Получаем список всех HDF5 файлов
    h5_files = [os.path.join(h5_directory, f) for f in os.listdir(h5_directory) 
               if f.endswith('.h5')]
    
    # Генерация якорей
    anchors = generate_anchors_from_h5_dataset(h5_files)
    
    # Сохранение в .mat формат (для совместимости с оригинальным кодом)
    savemat(output_mat_path, {'anchor_A': anchors, 'anchor_B': anchors})
    print(f"Якоря сохранены в {output_mat_path}")
    
    return anchors

# Пример использования:
if __name__ == '__main__':
    # Укажите путь к вашей директории с HDF5 файлами
    h5_data_dir = './data/'
    
    # Путь для сохранения якорей
    anchors_output_path = './resources/kaust_anchors.mat'
    
    # Создаем якоря
    custom_anchors = prepare_custom_anchors(h5_data_dir, anchors_output_path)