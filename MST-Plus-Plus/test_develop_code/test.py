import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
from skimage import exposure
import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
import physically_plausible as pp

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

sensitivity = pp.load_cie64cmf('../resources/', np.arange(400,701,9))

val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()

with open(f'{opt.data_root}/split_txt/test_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'

def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    vis_dir = os.path.join(opt.outf, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    all_mrae = []
    all_rmse = []
    all_psnr = []
    
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            if method=='awan':
                output = model(input[:, :, 118:-118, 118:-118])
                loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            else:
                output = model(input)
                images_rgb = input.permute(0, 2, 3, 1)
                funda_mat = torch.from_numpy(pp.calculate_fundamental_metamer(sensitivity)).float().cuda()
                output_spec = output.permute(0, 2, 3, 1)
                null_basis = torch.from_numpy(pp.calculate_null_basis(sensitivity)).float().cuda()
                output = (torch.matmul(images_rgb, funda_mat.permute(1, 0)) + torch.matmul(output_spec, null_basis.permute(1, 0))).permute(0, 3, 1, 2)
                loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
        all_mrae.append(loss_mrae.item())
        all_rmse.append(loss_rmse.item())
        all_psnr.append(loss_psnr.item())

        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)
        
        if i < 10: 
            visualize_results(input, target, output, result, mat_name, vis_dir, 
                            loss_mrae.item(), loss_rmse.item(), loss_psnr.item())
    
    plot_global_metrics(all_mrae, all_rmse, all_psnr, vis_dir)
    
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

def visualize_results(input, target, output, result, mat_name, vis_dir, mrae, rmse, psnr):
    input_img = input.cpu().numpy().squeeze().transpose(1, 2, 0)
    target_img = target.cpu().numpy().squeeze().transpose(1, 2, 0)
    output_img = output.cpu().numpy().squeeze().transpose(1, 2, 0)
    
    channels = [0, output_img.shape[2]//4, output_img.shape[2]//2, output_img.shape[2]-1]
    
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'{mat_name}\nMRAE: {mrae:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.4f}', y=1.02)
    
    plt.subplot(3, 4, 1)
    plt.imshow(input_img)
    plt.title('Input RGB')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    center_y, center_x = target_img.shape[0]//2, target_img.shape[1]//2
    plt.plot(target_img[center_y, center_x, :], label='Target')
    plt.plot(output_img[center_y, center_x, :], label='Output')
    plt.title('Central Pixel Spectrum')
    plt.legend()
    
    plt.subplot(3, 4, 3)
    diff = np.abs(target_img - output_img).mean(axis=2)
    plt.imshow(diff, cmap='hot', norm=PowerNorm(gamma=0.5))
    plt.colorbar()
    plt.title('Absolute Difference')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    sns.histplot(diff.flatten(), bins=50, kde=True)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    
    for i, channel in enumerate(channels):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(target_img[:, :, channel], cmap='viridis')
        plt.title(f'Target Channel {channel}')
        plt.axis('off')
        
        plt.subplot(3, 4, 9 + i)
        plt.imshow(output_img[:, :, channel], cmap='viridis')
        plt.title(f'Output Channel {channel}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'visualization_{mat_name.replace(".mat", ".png")}'), 
                bbox_inches='tight', dpi=150)
    plt.close()
    
    plot_pseudo_color(target_img, output_img, mat_name, vis_dir)

def plot_pseudo_color(target, output, mat_name, vis_dir):
    r, g, b = target.shape[2]//3, target.shape[2]//2, target.shape[2]//4
    
    target_rgb = target[:, :, [r, g, b]]
    output_rgb = output[:, :, [r, g, b]]
    
    target_rgb = exposure.rescale_intensity(target_rgb, out_range=(0, 1))
    output_rgb = exposure.rescale_intensity(output_rgb, out_range=(0, 1))
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(target_rgb)
    plt.title('Target Pseudo-RGB')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title('Output Pseudo-RGB')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'pseudo_color_{mat_name.replace(".mat", ".png")}'), 
                bbox_inches='tight', dpi=150)
    plt.close()

def plot_global_metrics(all_mrae, all_rmse, all_psnr, vis_dir):
    """Визуализация глобальных метрик по всем изображениям"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(all_mrae, 'b-')
    plt.xlabel('Image Index')
    plt.ylabel('MRAE')
    plt.title('MRAE across Images')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(all_rmse, 'r-')
    plt.xlabel('Image Index')
    plt.ylabel('RMSE')
    plt.title('RMSE across Images')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(all_psnr, 'g-')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR across Images')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'global_metrics.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_mrae, all_rmse, all_psnr], 
                labels=['MRAE', 'RMSE', 'PSNR'])
    plt.title('Distribution of Metrics')
    plt.grid(True)
    plt.savefig(os.path.join(vis_dir, 'metrics_distribution.png'), bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr = validate(val_loader, model)
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, psnr:{psnr}')