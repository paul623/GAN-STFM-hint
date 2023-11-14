import torch
from torch import nn
from torch.utils.data import DataLoader

from learning.SatelliteDataSet import SatelliteDataSet, Mode
from learning.scgan import Generator,Discriminator
from torchgan.losses import LeastSquaresDiscriminatorLoss,LeastSquaresGeneratorLoss
from pathlib import Path
from tqdm import tqdm

use_gpu = torch.cuda.is_available()
batch_size = 4
image_size = [6, 2720, 3200]
image_widith_height = [2720, 3200]
image_real_size = [6, 256, 256]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator(6, image_real_size)
discriminator = Discriminator(6, image_real_size)
train_dir = Path("/home/zbl/datasets/STFusion/LGC/LGC_data/train/")
val_dir = Path("/home/zbl/datasets/STFusion/LGC/LGC_data/val/")
PATCH_SIZE = 256
patch_stride = 200

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
g_loss = LeastSquaresGeneratorLoss()  # GAN的专用损失函数 生成器
d_loss = LeastSquaresDiscriminatorLoss()  # GAN的专用损失函数 辨别器

loss_fn = nn.MSELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

if use_gpu:
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    g_loss = g_loss.cuda()
    d_loss = d_loss.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

'''
#####################
train
######################
'''
generator.train()
discriminator.train()

print('Loading data...')
train_set = SatelliteDataSet(train_dir, image_widith_height, PATCH_SIZE, patch_stride, mode=Mode.TRAINING)
val_set = SatelliteDataSet(val_dir, image_widith_height, PATCH_SIZE, mode=Mode.VALIDATION)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=1, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=1)

num_epoch = 1
for epoch in range(num_epoch):
    print(f"-------------------------epoch:{epoch}-----------------------------------")
    for i, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')):
        data = [im.to(device) for im in data]
        inputs, target = data[:-1], data[-1]

        ############################
        # (1) Update D network 先更新判别器
        ###########################
        discriminator.zero_grad()
        generator.zero_grad()
        prediction = generator(inputs[0], inputs[1])
        # 待完善

