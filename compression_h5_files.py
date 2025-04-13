import os
import h5py
import numpy as np
from tqdm import tqdm 

def optimize_h5_file(input_path, output_path, compression="gzip", compression_opts=4):
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        def copy_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                f_out.create_dataset(
                    name,
                    data=obj[:],
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=True if obj.chunks is not None else None
                )
            elif isinstance(obj, h5py.Group):
                f_out.create_group(name)
        f_in.visititems(copy_item)

def process_folder(folder_path, compression="gzip", backup=False):
    h5_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    if not h5_files:
        print(f"Файлы .h5 не найдены в папке: {folder_path}")
        return

    print(f"Найдено файлов .h5: {len(h5_files)}")
    
    for file_path in tqdm(h5_files, desc="Оптимизация файлов"):
        try:
            temp_path = file_path + ".optimized"
        
            optimize_h5_file(file_path, temp_path, compression)
            
            if backup:
                backup_path = file_path + ".bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(file_path, backup_path)
            
            os.rename(temp_path, file_path)
            
        except Exception as e:
            print(f"\nОшибка при обработке файла {file_path}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Оптимизация .h5 файлов с сжатием.")
    parser.add_argument("folder", type=str, help="Путь к папке с файлами .h5")
    parser.add_argument("--compression", type=str, default="gzip", help="Метод сжатия (gzip, lzf)")
    parser.add_argument("--no-backup", action="store_false", help="Не создавать резервные копии")
    
    args = parser.parse_args()
    
    process_folder(
        folder_path=args.folder,
        compression=args.compression,
        backup=args.no_backup
    )   
