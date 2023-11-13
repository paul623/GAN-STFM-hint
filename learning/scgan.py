import torch
import torchvision
import torch.nn as nn
import numpy as np

'''
用的SCGAN
SCGAN是条件GAN，加入一个label标签，这里是一张参考图片
损失函数用的MSELoss
'''


class Generator(nn.Module):
    def __init__(self, in_channel, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(in_channel, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),

            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),

            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),

            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),  # prod计算数组中所有元素的乘积。 其实等价于1*28*28 花里胡哨的
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        output = self.model(z)
        image = output.reshape(z.shape[0], *self.image_size)

        return image


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            torch.nn.GELU(),
            nn.Linear(512, 256),
            torch.nn.GELU(),
            nn.Linear(256, 128),
            torch.nn.GELU(),
            nn.Linear(128, 64),
            torch.nn.GELU(),
            nn.Linear(64, 32),
            torch.nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, labels):
        # shape of image: [batchsize, 1, 28, 28]
        image = torch.cat([image.reshape(image.shape[0], -1), labels], dim=1)
        prob = self.model(image)

        return prob
