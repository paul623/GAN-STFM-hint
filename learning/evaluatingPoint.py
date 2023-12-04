"""
论文总共有MAE RMSE SAM SSIM四个指标
MAE：平均绝对误差 https://blog.csdn.net/qq_40206371/article/details/119814240
RMSE: 均方根误差 https://blog.csdn.net/qq_40206371/article/details/119814240
SAM: 光谱角相似度 https://blog.csdn.net/qq_36542241/article/details/109847722
SSIM:结构相似度 https://github.com/VainF/pytorch-msssim
"""
import numpy as np
import rasterio
import torch
from numpy.linalg import norm
from pytorch_msssim import ssim

ground_truth_dir = "/home/byt/Fusion/ganstfm-main/ganstfm-main/dataset/LGC_data/refs/20050129_TM.tif"
predict_dir = "/home/byt/Fusion/ganstfm-main/ganstfm-main/run_LGC_C2/test/PRED_2005_045_0214-2005_029_0129.tif"

def load_tiff_as_tensor(file_path):
    # 读取 TIFF 图像
    with rasterio.open(file_path) as src:
        image_data = src.read()
        image_data = image_data.transpose(1, 2, 0)  # 调整通道顺序
        return torch.from_numpy(image_data).unsqueeze(0).permute(0, 3, 1, 2).float()

def calculate_mae(image_pred, image_target):
    # 将张量转换为浮点型
    image_pred = image_pred.float()
    image_target = image_target.float()

    # 计算 MAE
    absolute_diff = torch.abs(image_pred - image_target)
    mae = torch.mean(absolute_diff)

    return mae.item()


def calculate_rmse(image_pred, image_target):
    # 将张量转换为浮点型
    image_pred = image_pred.float()
    image_target = image_target.float()

    # 计算 RMSE
    mse = torch.mean((image_pred - image_target) ** 2)
    rmse = torch.sqrt(mse)

    return rmse.item()

def calculate_sam(x_pred, x_true):
    # 初始化 SAM 结果
    sam_deg = 0.0
    total_pixels = 0

    # 读取预测和真实图像数据
    with rasterio.open(x_pred) as src_pred, rasterio.open(x_true) as src_true:
        # 读取数据并将其转换为浮点型
        image_data_pred = src_pred.read().astype(np.float32)
        image_data_true = src_true.read().astype(np.float32)

        # 调整通道顺序
        image_data_pred = np.transpose(image_data_pred, (1, 2, 0))
        image_data_true = np.transpose(image_data_true, (1, 2, 0))

        # 获取图像高度和宽度
        height, width, channels = image_data_pred.shape

        # 计算每个像素点的 SAM
        for i in range(height):
            for j in range(width):
                vec_pred = image_data_pred[i, j]
                vec_true = image_data_true[i, j]

                # 计算夹角
                sam_rad = np.arccos(np.dot(vec_pred, vec_true) / (norm(vec_pred) * norm(vec_true)))
                sam_deg += np.degrees(sam_rad)
                total_pixels += 1

    # 计算平均 SAM
    sam_deg /= total_pixels
    return sam_deg

def calculate_ssim(image_pred, image_target):
    return ssim(image_pred, image_target, data_range=255, size_average=False)


image_pred = load_tiff_as_tensor(predict_dir)
image_target = load_tiff_as_tensor(ground_truth_dir)


mae_value = calculate_mae(image_pred, image_target)
rmse_value = calculate_rmse(image_pred, image_target)
# sam_value = calculate_sam(predict_dir, ground_truth_dir)
ssim_value = calculate_ssim(image_pred, image_target)

print(f"Mean Absolute Error (MAE): {mae_value:.4f}")
print(f"Root Mean Square Error (RMSE): {rmse_value:.4f}")
# print(f"Spectral Angle Mapper(SAM): {sam_value:.4f}")
print(f"Structural Similarity Index (SSIM): {ssim_value.item():.4f}")
