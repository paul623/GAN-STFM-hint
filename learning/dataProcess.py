"""
@Project ：GAN-STFM-learning 
@File    ：dataProcess.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/11/6 18:15 
"""
# 用来处理图片以及加载数据集
import rasterio
from pathlib import Path

train_dir = Path("/home/zbl/datasets/STFusion/CIA/data_cia/train")
val_dir = Path("/home/zbl/datasets/STFusion/CIA/data_cia/val")
image_size = [3200, 2720]

tif_path = r"/home/zbl/datasets/STFusion/LGC/landsat_tif/20040416_TM.tif"

img = rasterio.open(tif_path)  # 读取影像
img_size = img.shape
img_bounds = img.bounds  # 影像四至
img_rows, img_cols = img.shape  # 影像行列号
img_bands = img.count  # 影像波段数
img_indexes = img.indexes  # 影像波段
img_crs = img.crs  # 影像坐标系
img_transform = img.transform  # 影像仿射矩阵
upper_left = img.transform * (0, 0)  # 影像左上角像元的左上角点经纬度
lower_right = img.transform * (img.width, img.height)  # 影像右下角像元的右下角点经纬度
img_meta = img.meta  # 影像基础信息，包含driver、数据类型、坐标系、仿射矩阵等
