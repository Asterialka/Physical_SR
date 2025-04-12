import torch
from .MST_Plus_Plus import MST_Plus_Plus


def model_generator(method, pretrained_model_path=None):
    if method == 'mst_plus_plus':
        model = MST_Plus_Plus().cpu()  # Модель сразу создаётся на CPU
    else:
        raise NameError(f'Method {method} is not defined!')  # Лучше вызывать ошибку
    
    if pretrained_model_path is not None:
        print(f'Loading model from {pretrained_model_path}')
        
        # Загружаем веса на CPU (даже если они сохранены на GPU)
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')  # 🔥 Ключевое изменение!
        
        # Удаляем 'module.', если модель сохранена в DataParallel/DDP
        state_dict = {
            k.replace('module.', ''): v 
            for k, v in checkpoint['state_dict'].items()
        }
        
        # Загружаем веса в модель
        model.load_state_dict(state_dict, strict=True)
    
    return model