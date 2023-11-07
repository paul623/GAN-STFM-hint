import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchgan.losses import LeastSquaresDiscriminatorLoss, LeastSquaresGeneratorLoss

from model import *
from data import PatchSet, Mode
from utils import *

import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.image_size = option.image_size

        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_d = self.train_dir / 'discriminator.pth'

        self.logger = get_logger()
        self.logger.info('Model initialization')

        self.generator = SFFusion().to(self.device)
        self.discriminator = MSDiscriminator().to(self.device)
        self.pretrained = AutoEncoder().to(self.device)
        load_pretrained(self.pretrained, 'assets/autoencoder.pth')

        # 评判标准
        self.criterion = ReconstructionLoss(self.pretrained)
        self.g_loss = LeastSquaresGeneratorLoss()   # GAN的专用损失函数 生成器
        self.d_loss = LeastSquaresDiscriminatorLoss()   # GAN的专用损失函数 辨别器

        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:  # 如果显卡大于一个，使用多个并行计算
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.discriminator = nn.DataParallel(self.discriminator, device_ids)

        # 设置优化器为Adam
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=option.lr)

        # 打印参数和模型结构
        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')
        n_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for discriminator.')
        self.logger.info(str(self.generator))
        self.logger.info(str(self.discriminator))

    def train_on_epoch(self, n_epoch, data_loader):
        self.generator.train()
        self.discriminator.train()
        epg_loss = AverageMeter()  # Computes and stores the average and current value
        epd_loss = AverageMeter()
        epg_error = AverageMeter()

        batches = len(data_loader)
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):  # 同时获取元素的索引和元素本身 直接写 for data in data_loader也行
            t_start = timer()
            data = [im.to(self.device) for im in data]  # 放cuda/cpu上运行
            '''
            data = [Ct, Ft', Ft] 即[要预测时刻的低分， 任意时刻的高分， 要预测时刻的高分]
            这里inputs取的是第一和第二项，即 inputs = [Ct, Ft'] 注意python里面切分是左闭右开
            这里target是第三项，即 target = [Ft]
            CGAN有两个输入，一个是噪声z一个是label，这里将Ft'经过特征网络提取特征后作为label输入，噪声z就是要预测的低分图像Ct
            '''
            inputs, target = data[:-1], data[-1]    # inputs =   target = [Ft]
            ############################
            # (1) Update D network 先更新判别器
            ###########################
            self.discriminator.zero_grad()
            self.generator.zero_grad()

            prediction = self.generator(inputs)  # inputs： [0, 1][batch_size, 6, 256, 256]
            '''
            prediction.detach() 创建一个新的张量，与张量共享权重，但是不参与梯度传播计算
            [input 1]: Ft Ct
            [input 2]: Ct prediction(生成的假 Ft)
            '''
            d_loss = (self.d_loss(self.discriminator(torch.cat((target, inputs[0]), 1)),
                                  self.discriminator(torch.cat((prediction.detach(), inputs[0]), 1))))
            d_loss.backward()
            self.d_optimizer.step()
            epd_loss.update(d_loss.item())
            ############################
            # (2) Update G network 再更新生成器
            ###########################
            '''
            self.criterion(prediction, target) 计算了生成器的损失
            self.g_loss是生成器专用loss LSGAN
            Ct 假Ft
            '''
            # a = self.criterion(prediction, target)
            # b = self.g_loss(self.discriminator(torch.cat((prediction, inputs[0]), 1)))
            # 生成器的损失函数以作者自定义的为准 辅之判别器的损失函数，discriminator使用的是gan封装好的LeastSquaresDiscriminatorLoss, MSELoss与1tensor做差值
            g_loss = (self.criterion(prediction, target) + 1e-3 *
                      self.g_loss(self.discriminator(torch.cat((prediction, inputs[0]), 1))))
            g_loss.backward()
            self.g_optimizer.step()

            epg_loss.update(g_loss.item())
            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'G-Loss: {g_loss.item():.6f} - '
                             f'D-Loss: {d_loss.item():.6f} - '
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.discriminator, self.d_optimizer, self.last_d)
        return epg_loss.avg, epd_loss.avg, epg_error.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        self.discriminator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs, target = data[:-1], data[-1]
            prediction = self.generator(inputs)
            g_loss = F.mse_loss(prediction, target)
            epoch_error.update(g_loss.item())
        return epoch_error.avg

    def train(self, train_dir, val_dir, patch_stride, batch_size,
              num_workers=0, epochs=50, resume=True):
        last_epoch = -1
        least_error = float('inf')
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer) # 加载上次训练保存的权重
            load_checkpoint(self.last_d, self.discriminator, optimizer=self.d_optimizer)
        start_epoch = last_epoch + 1

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
        val_set = PatchSet(val_dir, self.image_size, PATCH_SIZE, mode=Mode.VALIDATION)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        self.logger.info('Training...')
        for epoch in range(start_epoch, epochs + start_epoch):
            self.logger.info(f"Learning rate for Generator: "
                             f"{self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: "
                             f"{self.d_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_d_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss', 'train_d_loss', 'train_g_error', 'val_error']
            csv_values = [epoch, train_g_loss, train_d_loss, train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
        self.logger.info('Testing...')
        # 训练的时候可以通过不同步长来切割重叠块以提供更多信息用于分析或者处理
        # 预测/测试的时候就可以直接按照块大小来切分
        assert self.image_size[0] % patch_size[0] == 0
        assert self.image_size[1] % patch_size[1] == 0
        rows = int(self.image_size[1] / patch_size[1])
        cols = int(self.image_size[0] / patch_size[0])
        n_blocks = rows * cols
        image_dirs = iter([p for p in test_dir.iterdir() if p.is_dir()])
        test_set = PatchSet(test_dir, self.image_size, patch_size, mode=Mode.PREDICTION)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        pixel_scale = 10000
        patches = []
        t_start = timer()
        count = 1
        for inputs in test_loader:
            print(f"count:{count} in {n_blocks}")
            inputs = [im.to(self.device) for im in inputs]
            prediction = self.generator(inputs)
            prediction = prediction.squeeze().cpu().numpy()
            patches.append(prediction * pixel_scale)
            count = count+1
            # 完成一张影像以后进行拼接
            if len(patches) == n_blocks:
                count = 1
                print("完成一张。。。")
                result = np.empty((NUM_BANDS, *self.image_size), dtype=np.float32)
                block_count = 0
                for i in range(rows):
                    row_start = i * patch_size[1]
                    for j in range(cols):
                        col_start = j * patch_size[0]
                        result[:,
                        col_start: col_start + patch_size[0],
                        row_start: row_start + patch_size[1]
                        ] = patches[block_count]
                        block_count += 1
                patches.clear()
                # 存储预测影像结果
                result = result.astype(np.int16)
                metadata = {
                    'driver': 'GTiff',
                    'width': self.image_size[1],
                    'height': self.image_size[0],
                    'count': NUM_BANDS,
                    'dtype': np.int16
                }
                name = f'PRED_{next(image_dirs).stem}.tif'
                save_array_as_tif(result, self.test_dir / name, metadata)
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s on {name}')
                t_start = timer()
