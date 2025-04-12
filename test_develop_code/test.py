 import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR
from hsi_dataset import ValidDataset
from torch.utils.data import DataLoader
import physically_plausible as pp
import matplotlib.pyplot as plt
import seaborn as sns

# Убираем предупреждения
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# =======================
#         ARGPARSE
# =======================
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../dataset/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# Создание папки
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# =======================
#      ЗАГРУЗКА ДАННЫХ
# =======================
sensitivity = pp.load_cie64cmf('../resources/', np.arange(400,701,9))

val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# =======================
#        LOSS
# =======================
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# =======================
# Визуализация
# =======================
def visualize_prediction(input_rgb, target, output, idx, vis_dir):
    input_rgb = input_rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    target = target.squeeze().permute(1, 2, 0).cpu().numpy()
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()

    input_rgb = np.clip(input_rgb, 0, 1)
    target_rgb = np.clip(np.mean(target, axis=2), 0, 1)
    pred_rgb = np.clip(np.mean(output, axis=2), 0, 1)

    # Сравнительные визуализации
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    axs[0].imshow(input_rgb)
    axs[0].set_title("Input RGB")
    axs[1].imshow(target_rgb, cmap='gray')
    axs[1].set_title("Target HS (avg)")
    axs[2].imshow(pred_rgb, cmap='gray')
    axs[2].set_title("Predicted HS (avg)")
    axs[3].imshow(np.abs(pred_rgb - target_rgb), cmap='hot')
    axs[3].set_title("Difference")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f"Sample #{idx}")
    plt.tight_layout()
    
    # Сохранение
    fig_path = os.path.join(vis_dir, f"sample_{idx:03d}_comparison.png")
    plt.savefig(fig_path)
    plt.close()

    # Спектральный профиль
    h, w, _ = target.shape
    center_h, center_w = h // 2, w // 2
    plt.figure(figsize=(8, 4))
    plt.plot(target[center_h, center_w, :], label='GT Spectrum')
    plt.plot(output[center_h, center_w, :], label='Predicted Spectrum')
    plt.title(f"Spectral Profile at Center Pixel (Sample #{idx})")
    plt.xlabel("Spectral Band")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.grid(True)

    # Сохранение
    spec_path = os.path.join(vis_dir, f"sample_{idx:03d}_spectrum.png")
    plt.savefig(spec_path)
    plt.close()

def plot_metrics(all_mrae, all_rmse, all_psnr, vis_dir):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    sns.histplot(all_mrae, bins=20, kde=True)
    plt.title('MRAE Distribution')

    plt.subplot(1, 3, 2)
    sns.histplot(all_rmse, bins=20, kde=True)
    plt.title('RMSE Distribution')

    plt.subplot(1, 3, 3)
    sns.histplot(all_psnr, bins=20, kde=True)
    plt.title('PSNR Distribution')

    plt.tight_layout()
    metrics_path = os.path.join(vis_dir, "metrics_distribution.png")
    plt.savefig(metrics_path)
    plt.close()

# =======================
#       VALIDATION
# =======================
with open(f'{opt.data_root}/split_txt/test_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
var_name = 'cube'

def validate(val_loader, model):
    vis_dir = os.path.join(opt.outf, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    all_mrae_list = []
    all_rmse_list = []
    all_psnr_list = []

    for i, (input, target) in enumerate(val_loader):
        input = input
        target = target

        with torch.no_grad():
            if method == 'awan':
                output = model(input[:, :, 118:-118, 118:-118])
                loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            else:
                output = model(input)
                images_rgb = input.permute(0, 2, 3, 1)
                funda_mat = torch.from_numpy(pp.calculate_fundamental_metamer(sensitivity)).float()
                output_spec = output.permute(0, 2, 3, 1)
                null_basis = torch.from_numpy(pp.calculate_null_basis(sensitivity)).float()
                output = (torch.matmul(images_rgb, funda_mat.permute(1, 0)) +
                          torch.matmul(output_spec, null_basis.permute(1, 0))).permute(0, 3, 1, 2)

                loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])

        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

        all_mrae_list.append(loss_mrae.item())
        all_rmse_list.append(loss_rmse.item())
        all_psnr_list.append(loss_psnr.item())

        # Сохраняем результат
        result = output.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.clip(result, 0, 1)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)

        # Визуализация первых 5
        if i < 5:
            visualize_prediction(input.cpu(), target.cpu(), output.cpu(), i, vis_dir)

    # Построение распределения метрик
    plot_metrics(all_mrae_list, all_rmse_list, all_psnr_list, vis_dir)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

# =======================
#          MAIN
# =======================
if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).to('cpu')

    mrae, rmse, psnr = validate(val_loader, model)
    print(f'\n✅ Method: {method}\nMRAE: {mrae:.4f}\nRMSE: {rmse:.4f}\nPSNR: {psnr:.2f} dB') 