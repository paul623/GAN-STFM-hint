"""
@Project ：GAN-STFM-learning 
@File    ：dataProcess.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/11/6 18:15 
"""
# 用来处理图片以及加载数据集

import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import rasterio
from torch.utils.data import DataLoader, Dataset
from original import experiment
from pathlib import Path
from original.data import PatchSet, Mode


train_dir = Path("/home/zbl/datasets/STFusion/CIA/data_cia/train")
val_dir = Path("/home/zbl/datasets/STFusion/CIA/data_cia/val")
image_size = [3200, 2720]

train_set = PatchSet(train_dir, image_size, 256, 200, mode=Mode.TRAINING)
print(train_set)
# class RemoteSensingDataset(Dataset):
#     def __init__(self, data_dir, path_size, transform=None):
#         self.data_dir = data_dir
#         self.image_files = [f for f in os.listdir(data_dir) if f.is_dir()]
#         self.path_size = path_size
#         self.transform = transform

