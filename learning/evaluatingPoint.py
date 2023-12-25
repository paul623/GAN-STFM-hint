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
import tifffile
from pytorch_msssim import ssim
from sewar import rmse, ssim, sam, psnr, mse
ground_truth_dir ="/home/zbl/datasets/STFusion/CIA/data_cia/refs/20020221_TM.tif"
predict_dir = "/home/zbl/datasets/STFusion/RunLog/baseline/CIA/test/PRED_2002_069_0309-2002_053_0221.tif"

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


# def calculate_rmse(image_pred, image_target):
#     # 将张量转换为浮点型
#     image_pred = image_pred.float()
#     image_target = image_target.float()
#
#     # 计算 RMSE
#     mse = torch.mean((image_pred - image_target) ** 2)
#     rmse = torch.sqrt(mse)
#
#     return rmse.item()
#
# def spectral_angle_mapper(spectrum1, spectrum2):
#
#     # 归一化两个光谱向量
#     spectrum1_normalized = spectrum1 / np.linalg.norm(spectrum1)
#     spectrum2_normalized = spectrum2 / np.linalg.norm(spectrum2)
#
#     # 计算夹角余弦值
#     cosine_angle = np.dot(spectrum1_normalized, spectrum2_normalized)
#
#     # 计算夹角
#     angle = np.arccos(cosine_angle)
#
#     # 将弧度转换为度
#     angle_degrees = np.degrees(angle)
#
#     return angle_degrees
#
# def calculate_sam(image1_path, image2_path):
#     # 通过 tifffile 读取 TIFF 格式图像
#     image1 = tifffile.imread(image1_path)
#     image2 = tifffile.imread(image2_path)
#
#     # 获取图像的光谱向量（每个像素点）
#     spectra1 = image1.reshape((-1, image1.shape[-1]))
#     spectra2 = image2.reshape((-1, image2.shape[-1]))
#
#
#     # 初始化 SAM 值
#     sam_values = []
#
#     # 计算 SAM
#     for spec1, spec2 in zip(spectra1, spectra2):
#         sam = spectral_angle_mapper(spec1, spec2)
#         sam_values.append(sam)
#
#     # 计算平均 SAM
#     average_sam = np.mean(sam_values)
#
#     return average_sam
#
#
# def calculate_ssim(image_pred, image_target):
#     return ssim(image_pred, image_target, data_range=255, size_average=False)



# image_pred = load_tiff_as_tensor(predict_dir)
# image_target = load_tiff_as_tensor(ground_truth_dir)


# mae_value = calculate_mae(image_pred, image_target)
# rmse_value = calculate_rmse(image_pred, image_target)
# sam_value = calculate_sam(predict_dir, ground_truth_dir)
# ssim_value = calculate_ssim(image_pred, image_target)

# 读取图像
image_pred = tifffile.imread(predict_dir)
image_target = tifffile.imread(ground_truth_dir)

mae_value = calculate_mae(load_tiff_as_tensor(predict_dir), load_tiff_as_tensor(ground_truth_dir))
print(f"Mean Absolute Error (MAE): {mae_value:.4f}")
rmse_value = rmse(image_pred, image_target)
print(f"Root Mean Square Error (RMSE): {rmse_value:.4f}")
sam_value = sam(image_pred, image_target)
print(f"Spectral Angle Mapper(SAM): {sam_value:.4f}")
ssim_value = ssim(image_pred, image_target)
print(f"Structural Similarity Index (SSIM): {ssim_value[0]:.4f}")





