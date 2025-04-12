import torch
from .MST_Plus_Plus import MST_Plus_Plus

def model_generator(method, pretrained_model_path=None):
    if method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
