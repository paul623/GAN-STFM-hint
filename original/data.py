from configparser import Interpolation
from pathlib import Path
import numpy as np
import rasterio
import math
from enum import Enum, auto, unique
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import make_tuple

'''
2023年12月3日
这个文件主要是用来对遥感图像进行切块处理
定义了一个PatchSet，可以直接当成dataset来用
只需要注意一下patch_size大小，在训练的时候无所谓，但是测试的时候必须都能被高宽整除
'''


@unique
class Mode(Enum):
    TRAINING = auto()
    VALIDATION = auto()
    PREDICTION = auto()


def get_pair_path(directory: Path, mode: Mode):
    paths: list = [None] * 3
    if mode is Mode.TRAINING:
        year, coarse, fine = directory.name.split('_')
        for f in directory.glob('*.tif'):
            paths[0 if year + coarse in f.name else 2] = f  # 两张图片， landsat高分放2，modis低分放0
        refs = [p for p in (directory.parents[1] / 'refs').glob('*.tif')]  # 加载所有refs参考
        paths[1] = refs[np.random.randint(0, len(refs))]  # 随机选取一张放入1中
    else:
        ref_label, pred_label = directory.name.split('-')
        ref_tokens, pred_tokens = ref_label.split('_'), pred_label.split('_')
        for f in directory.glob('*.tif'):
            order = {
                pred_tokens[0] + pred_tokens[1] in f.name: 0,
                ref_tokens[0] + ref_tokens[2] in f.name: 1,
                pred_tokens[0] + pred_tokens[2] in f.name: 2
            }
            if True in order.keys():
                paths[order[True]] = f
        if mode is Mode.PREDICTION:
            del paths[2]
    return paths


# 按照get_pair_path返回的三张图片路径来加载图像
def load_image_pair(directory: Path, mode: Mode):
    paths = get_pair_path(directory, mode=mode)
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            images.append(im)
    return images


def crop_and_resize_image_global(image, id_x, id_y, patch_size):
    # 裁剪成 patch_size 的两倍
    # cropped_image = image[:, id_x: (id_x + patch_size[0] * 2), id_y: (id_y + patch_size[1] * 2)]
    cropped_image = image[:, id_x: (id_x + 50), id_y: (id_y + 50)]
    cropped_image = PatchSet.transform(cropped_image)
    # 调整大小至 patch_size
    resized_image = TF.resize(cropped_image, patch_size, interpolation=Image.BILINEAR)
    return resized_image


def process_images_global(images, id_x, id_y, patch_size):
    processed_images = []
    for img in images:
        processed_img = crop_and_resize_image_global(img, id_x, id_y, patch_size)
        processed_images.append(processed_img)
    return processed_images


class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None, mode=Mode.TRAINING):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size
        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.mode = mode

        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)
        # 计算出图像进行分块以后的patches的数目 math.ceil向上取整 size + n(stride) = x + 1
        self.n_patch_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])  # patch_size = [256,256]
        self.n_patch_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])  # path_stride = [200, 200]
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y

    @staticmethod
    def transform(data):
        data[data < 0] = 0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        out = data.mul_(0.0001)
        return out

    def map_index(self, index):
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_y)
        return id_n, id_x, id_y


    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)

        images = load_image_pair(self.image_dirs[id_n], mode=self.mode)

        patches = [None] * (len(images))

        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0]),
                 id_y: (id_y + self.patch_size[1])]
            patches[i] = self.transform(im)

        global_branch = process_images_global(images=images[:2], id_x=id_x, id_y=id_y, patch_size=self.patch_size)
        # patches.append(TF.resize(images[0],(80, 80)))  # 要预测时刻的低分
        # patches.append(TF.resize(images[1],(80, 80)))  # 任意时刻的高分

        del images[:]
        del images
        return patches, global_branch

    def __len__(self):
        return self.num_patch
