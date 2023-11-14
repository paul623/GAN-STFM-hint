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
    def __init__(self, channels, image_size):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Conv2d(channels * 2, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 修正此处为 nn.Tanh()
        )

    def forward(self, x1, x2):
        # 在通道维度上拼接两张图像
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Conv2d(channels * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 在通道维度上拼接两张图像
        x = torch.cat([x1, x2], dim=1)
        return self.model(x)