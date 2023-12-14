import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from ssim import msssim
from normalization import SwitchNorm2d

'''
模型定义部分
主要是CGAN的结构
使用AutoEncoder调用了预训练权重，来帮助网络训练
'''


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()


NUM_BANDS = 6  # 图片通道数
PATCH_SIZE = 256  # 切块大小
SCALE_FACTOR = 16  # 裁剪尺度


# 上采样
class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        # 默认使用双线性插值方法，以在缩放时平滑地估计像素值
        # 例如，scale_factor=2 表示在每个维度上将尺寸扩大两倍，scale_factor=(0.5, 0.5) 表示在每个维度上将尺寸缩小为原来的一半。
        return F.interpolate(inputs, scale_factor=self.scale_factor)


'''
生成器的损失，主要是Feature，Spectrum光谱，Vision视觉，GAN四个损失

均方误差（MSE）余弦相似度（Cosine Similarity） 结构相似性指数（MS-SSIM）
余弦相似度衡量了两个向量之间的方向一致性，取值范围在 -1 到 1 之间，1 表示完全相似，-1 表示完全相反。
MSSIM是参考自https://github.com/jorge-pessoa/pytorch-msssim 多尺度结构相似度
这里传入了某个模型，然后利用这个模型的卷积层来更好的提取特征再来计算损失

'''


class ReconstructionLoss(nn.Module):
    def __init__(self, model, alpha=1.0, beta=1.0, gamma=1.0):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            self.model.conv1,
            self.model.conv2,
            self.model.conv3,
            self.model.conv4
        )

    def forward(self, prediction, target):
        _prediction, _target = self.encoder(prediction), self.encoder(target)
        loss = (self.alpha * F.mse_loss(_prediction, _target) +
                self.gamma * (1.0 - torch.mean(F.cosine_similarity(_prediction, _target, 1))) +
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        return loss


'''
本质上重写了Conv2d，设置kernel大小为3，padding为1
'''


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


'''
增加了一个ReplicationPad2d
'''


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            # 这个层通常在卷积神经网络中用来调整输入图像的大小，以确保卷积操作后输出与输入的尺寸相匹配。
            # 1代表在每一个边界上复制1个像素的宽度和高度填充
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


'''
卷积块
'''


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []

        if sampling == Sampling.DownSampling:  # 如果是下采样
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        else:
            if sampling == Sampling.UpSampling:   # 如果是上采样
                layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)

'''
GEncoder ResBlock
生成器的编码器
加入可变归一化的 残差快
批归一化通过对每层的输出进行归一化处理，有助于提高训练的稳定性、加速收敛速度，并有助于防止过拟合。
'''

class ResidulBlockWtihSwitchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlockWtihSwitchNorm, self).__init__()
        channels = min(in_channels, out_channels) # 取输入输出通道的最小值
        residual = [  # 对Fine Reference参考进行特征提取
            SwitchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            Conv3X3WithPadding(in_channels, channels),  # 其实就是多了一个nn.ReplicationPad2d(1)
            SwitchNorm2d(channels),  # 可以选择多种归一化
            nn.LeakyReLU(inplace=True),
            # 比论文多了一层卷积，为了调整输出通道
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [ # 对Coarse on prediction要预测的粗糙图像进行特征提取
            Conv3X3WithPadding(in_channels, channels),
            nn.Conv2d(channels, out_channels, 1),
            nn.LeakyReLU(inplace=True)
        ]
        if sampling == Sampling.UpSampling:  # 如果进行上采样
            residual.insert(2, Upsample(2))  # 在Conv3之前插入上采样模块
            transform.insert(0, Upsample(2))  # 在提取特征之后，主要是因为该图片是粗糙的，直接上采样效果不好
        elif sampling == Sampling.DownSampling:  # 如果进行下采样
            residual[2] = Conv3X3WithPadding(in_channels, channels, 2)  # 在Conv3之前插入上采样模块
            transform[0] = Conv3X3WithPadding(in_channels, channels, 2)     # 在提取特征之前进行下采样操作

        self.residual = nn.Sequential(*residual)
        self.transform = nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs[1])  # 细粒度的特征Fine reference
        lateral = self.transform(inputs[0])  # 输入的是coarse on prediction 粗粒度特征
        if len(inputs) == 4:
            global_prdict = self.residual(inputs[2])
            global_ref = self.residual(inputs[3])
            return lateral, trunk + lateral + global_prdict + global_ref, global_prdict, global_ref  # lateral:Coarse Features  trunk+lateral: Adjusted fine features
        else:
            return lateral, trunk + lateral


'''
GDecoder ResBlock
生成器的解码器
输出是 Concatenated features
输出是 Adjusted fine features
'''

class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sampling=None):
        super(ResidulBlock, self).__init__()
        channels = min(in_channels, out_channels)
        residual = [
            Conv3X3WithPadding(in_channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, out_channels, 1)
        ]
        transform = [nn.Conv2d(in_channels, out_channels, 1)]

        if sampling == Sampling.UpSampling:
            residual.insert(0, Upsample(2))
            transform.insert(0, Upsample(2))
        elif sampling == Sampling.DownSampling:
            residual[0] = Conv3X3WithPadding(in_channels, channels, 2)
            transform[0] = nn.Conv2d(in_channels, out_channels, 1, stride=2)

        self.residual = nn.Sequential(*residual)
        self.transform = transform[0] if len(transform) == 1 else nn.Sequential(*transform)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        lateral = self.transform(inputs)
        return trunk + lateral


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling)
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        out = self.conv8(torch.cat((l1, l7), 1))
        return out

'''
Generator
两部分组成，encoder和decoder
encoder使用的是带可变归一化的残差块
decoder使用的是普通的残差块
这里使用的都是3*3的卷积块哦~
'''
class SFFusion(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        channels = (16, 32, 64, 128)
        super(SFFusion, self).__init__()
        self.encoder = nn.Sequential(
            ResidulBlockWtihSwitchNorm(in_channels, channels[0]),
            ResidulBlockWtihSwitchNorm(channels[0], channels[1]),
            ResidulBlockWtihSwitchNorm(channels[1], channels[2]),
            ResidulBlockWtihSwitchNorm(channels[2], channels[3])  # 输出的是两张图片， 粗粒度图像和融合图像
        )
        self.decoder = nn.Sequential(
            ResidulBlock(channels[3] * 2, channels[3]),
            ResidulBlock(channels[3], channels[2]),
            ResidulBlock(channels[2], channels[1]),
            ResidulBlock(channels[1], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1)
        )

    def forward(self, inputs):  # inputs实际上是两张图片，所以需要把他们拼成一个
        x = self.encoder(inputs)[:2]
        return self.decoder(torch.cat(x, 1))  # 按照通道数拼在一起，因此通道数变成两倍

'''
D-ResBlock
判别器的残差块
'''
class ResidulBlockWithSpectralNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlockWithSpectralNorm, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(  # 旨在提升GAN中的判别器的表现和稳定性
                Conv3X3NoPadding(in_channels, in_channels, stride=2)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(
                nn.Conv2d(in_channels, out_channels, 1)),  # 调整通道
        )
        self.transform = SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, stride=2))

    def forward(self, inputs):
        return self.transform(inputs) + self.residual(inputs)


# 判别器网络
class Discriminator(nn.Sequential):
    def __init__(self, channels):
        modules = []
        for i in range(1, (len(channels))):
            modules.append(ResidulBlockWithSpectralNorm(channels[i - 1], channels[i]))
        modules.append(SpectralNorm2d(nn.Conv2d(channels[-1], 1, 1)))
        super(Discriminator, self).__init__(*modules)

    def forward(self, inputs):
        prediction = super(Discriminator, self).forward(inputs)
        # prediction = F.sigmoid(prediction)      # 2023年12月3日 add
        return prediction.view(-1, 1).squeeze(1)  # 将输出的形状从 (batch_size, 1, h, w) 转换为 (batch_size, 1) 的形状，以适应二分类任务的需要

'''
Multi scale Discriminator
'''

class MSDiscriminator(nn.Module):
    def __init__(self):
        super(MSDiscriminator, self).__init__()
        self.d1 = Discriminator((NUM_BANDS * 2, 32, 32, 64, 64, 128, 128, 256, 256))  # 9个残差块
        self.d2 = Discriminator((NUM_BANDS * 2, 32, 64, 64, 128, 128, 256, 256))  # 8个残差块
        self.d3 = Discriminator((NUM_BANDS * 2, 32, 64, 128, 128, 256, 256))  # 7个残差块

    def forward(self, inputs):
        l1 = self.d1(inputs)
        l2 = self.d2(F.interpolate(inputs, scale_factor=0.5))  # F.interpolate是插值操作，对输入进行0.5倍缩放插值
        l3 = self.d3(F.interpolate(inputs, scale_factor=0.25))  # 对输入进行0.25倍缩放
        return torch.mean(torch.stack((l1, l2, l3)))  # stack是堆叠的意思，即由原来的尺度变成[3,原来形状]
